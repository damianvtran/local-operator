"""Access ends at reconcile: a lost share drops the grant and says so.

THE DESIGN SENTENCE THIS PINS (§4): "a removed member is pruned from placement at
removal (owner) and fails the handshake on its next contact (member); its cached
grants are dropped when the next reconcile observes it is no longer a holder.
Access ends at reconcile, not at token expiry."

Three cells, one per actor:

* the MEMBER drops its cached grant when placement no longer names it a holder,
  records ``revoked``, and then refuses a fresh borrow LOCALLY (no dial);
* the MEMBER drops everything when its own membership row is gone or inactive —
  the removed-device half;
* the OWNER prunes the removed device out of every placement entry at removal,
  which is what makes the member's next pull converge.
"""

from __future__ import annotations

import time
from pathlib import Path

from local_operator.network import relay, store, types, wire
from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import mcp_state, placement as placement_mod
from local_operator.network.credentials.types import (
    CredentialRef,
    Grant,
    credential_key_for_mcp,
)

SELF = "d_" + "1" * 32
PEER = "d_" + "2" * 32
NETWORK = "n_" + "a" * 24
URL = "https://mcp.example.test/mcp"
KEY = credential_key_for_mcp(URL)


def _record(*, self_active: bool = True) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        epoch=1,
    )
    record.members.append(
        types.MemberRecord(
            device_id=SELF,
            public_key=wire.b64u(b"a" * 32),
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="self",
            lifecycle="active" if self_active else "expired",
            removed_at=None if self_active else 1.0,
        )
    )
    record.members.append(
        types.MemberRecord(
            device_id=PEER,
            public_key=wire.b64u(b"b" * 32),
            role="drive",
            capabilities=sorted(types.capabilities_for_role("drive")),
            added_via="invite",
        )
    )
    return record


def _place(root: Path, *, holder: bool = True) -> None:
    with placement_mod.mutate(NETWORK, root=root, self_device=PEER) as document:
        document.declare(
            KEY,
            owner_device=PEER,
            owner_device_name="macbook",
            provider="mcp-oauth",
            kind="mcp-rotating",
            by=PEER,
        )
        if holder:
            document.grant(KEY, SELF, by=PEER)


def _grant() -> Grant:
    now_ms = int(time.time() * 1000)
    return Grant(
        access_token="stub" + "-bearer",
        kind="bearer",
        token_expires_at_ms=now_ms + 600_000,
        grant_expires_at_ms=now_ms + 600_000,
        credential_ref=CredentialRef(
            owner_device=PEER,
            owner_device_name="macbook",
            provider="mcp-oauth",
            kind="mcp-oauth",
            credential_id=7,
        ),
        served_by=PEER,
        grant_id="g-1",
    )


def _client(root: Path) -> client_mod.MeshCredentialClient:
    from local_operator.network.credentials.placement import PlacementDocument

    return client_mod.MeshCredentialClient(
        root=root,
        self_device=SELF,
        network_id=NETWORK,
        placement=PlacementDocument.load(NETWORK, root, self_device=SELF),
    )


def test_a_lost_holder_row_drops_the_grant_and_records_revoked(root: Path) -> None:
    store.save(_record(), root)
    _place(root)
    client = _client(root)
    client.grants.put(KEY, "", _grant())

    # The owner revokes the share: the member's document no longer names it.
    with placement_mod.mutate(NETWORK, root=root, self_device=PEER) as document:
        document.revoke(KEY, SELF, by=PEER)

    dropped = client.revoke_local_grants([KEY])

    assert dropped == [KEY]
    assert client.grants.peek(KEY, "") is None
    evidence = mcp_state.load(NETWORK, root)[KEY]
    assert evidence["state"] == "revoked"
    assert evidence["code"] == "not_a_holder"

    refusal = client.request_grant_sync(KEY)
    assert getattr(refusal, "code", "") == "not_a_holder", refusal


def test_an_inactive_self_member_drops_every_grant(root: Path) -> None:
    store.save(_record(self_active=False), root)
    _place(root)
    client = _client(root)
    client.grants.put(KEY, "", _grant())

    dropped = client.revoke_local_grants()

    assert dropped == [KEY]
    assert mcp_state.load(NETWORK, root)[KEY]["code"] == "member_not_active"


def test_remove_member_prunes_the_removed_device_from_placement(root: Path) -> None:
    record = _record()
    store.save(record, root)
    _place(root)
    state = types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32))

    relay.remove_member(record, state, device_id=PEER, by=SELF, root=root)

    document = placement_mod.PlacementDocument.load(NETWORK, root)
    entry = document.entry(KEY)
    # The entry was the PEER's: its owner left, so the row goes entirely — a
    # holder-list edit would leave a share nobody can serve.
    assert entry is None


def test_remove_member_prunes_a_holder_without_deleting_the_row(root: Path) -> None:
    record = _record()
    store.save(record, root)
    with placement_mod.mutate(NETWORK, root=root, self_device=SELF) as document:
        document.declare(
            KEY,
            owner_device=SELF,
            owner_device_name="laptop",
            provider="mcp-oauth",
            kind="mcp-rotating",
            by=SELF,
        )
        document.grant(KEY, PEER, by=SELF)
    state = types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32))

    relay.remove_member(record, state, device_id=PEER, by=SELF, root=root)

    entry = placement_mod.PlacementDocument.load(NETWORK, root).entry(KEY)
    assert entry is not None
    assert entry.is_holder(SELF)
    assert not entry.is_holder(PEER)


def test_a_cold_not_a_holder_is_not_a_revocation(root: Path) -> None:
    """Requirement-6 inversion guard: a server nobody shared is not "revoked".

    A just-mirrored server refuses locally with ``not_a_holder`` (no holder row,
    no dial). Recording that as ``revoked`` made the status claim access had been
    WITHDRAWN for a server that never had any — the opposite of
    ``not_yet_synced``/``needs_operator_login``. Only a prior claim can be lost.
    """
    store.save(_record(), root)
    _place(root, holder=False)

    refusal = _client(root).request_grant_sync(KEY)
    assert getattr(refusal, "code", "") == "not_a_holder"
    assert mcp_state.load(NETWORK, root) == {}, "a cold refusal wrote evidence"

    client = _client(root)
    client._remember_evidence(  # noqa: SLF001 — the rule under test
        KEY,
        client_mod.BrokerError(code="not_a_holder", key=KEY, owner_device=PEER),
    )
    assert mcp_state.load(NETWORK, root) == {}

    # With a grant in hand, the same code IS a revocation.
    client.grants.put(KEY, "", _grant())
    client._remember_evidence(  # noqa: SLF001
        KEY,
        client_mod.BrokerError(code="not_a_holder", key=KEY, owner_device=PEER),
    )
    assert mcp_state.load(NETWORK, root)[KEY]["state"] == "revoked"
