"""Membership: admission, revocation, rotation, reconcile and the panic rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network import identity, relay, store, types, wire

NETWORK = "n_0123456789abcdef01234567"
SELF = "d_" + "a" * 32
PEER = "d_" + "b" * 32


def _record(root: Path | None = None) -> tuple[types.NetworkRecord, types.SecretState]:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=1,
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    record.members.append(
        types.MemberRecord(
            device_id=SELF,
            public_key=wire.b64u(b"a" * 32),
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="self",
        )
    )
    state = types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32))
    return record, state


def _admit_peer(record: types.NetworkRecord, *, role: str = "drive") -> types.MemberRecord:
    return relay.admit(
        record,
        device_id=PEER,
        public_key=wire.b64u(b"b" * 32),
        name="laptop",
        role=role,
        capabilities=sorted(types.capabilities_for_role(role)),
        added_by=SELF,
        added_via="invite",
        root=None,
        persist=False,
    )


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_admission_stores_the_resolved_capability_set() -> None:
    """Stored, not derived at read time: a later change to ``ROLE_CAPABILITIES``
    must not retroactively widen an existing member."""
    record, _state = _record()
    member = _admit_peer(record, role="drive")
    assert set(member.capabilities) == set(types.capabilities_for_role("drive"))
    assert "delete" not in member.capabilities
    del record  # the row is the thing under test


def test_a_burned_id_can_never_be_admitted_again() -> None:
    """R5's teeth: a removed device must not be able to wait for a re-admission."""
    record, _state = _record()
    record.removed_ids.append(PEER)
    with pytest.raises(types.MeshRefusal) as excinfo:
        _admit_peer(record)
    assert excinfo.value.code == "device_id_conflict"
    assert record.member(PEER) is None


def test_the_same_id_with_a_different_key_is_refused() -> None:
    """An id is a name; two keys claiming one name is either a collision or an
    attack, and neither may be absorbed silently."""
    record, _state = _record()
    _admit_peer(record)
    with pytest.raises(types.MeshRefusal) as excinfo:
        relay.admit(
            record,
            device_id=PEER,
            public_key=wire.b64u(b"c" * 32),
            role="read",
            root=None,
            persist=False,
        )
    assert excinfo.value.code == "device_id_conflict"


def test_re_admitting_the_same_row_is_idempotent() -> None:
    record, _state = _record()
    _admit_peer(record)
    again = _admit_peer(record)
    assert len(record.members) == 2
    assert again.device_id == PEER


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


def test_removing_a_member_tombstones_rotates_and_bumps_the_epoch(root: Path) -> None:
    record, state = _record()
    _admit_peer(record)
    store.save(record, root)
    store.save_secrets(state, root)
    outcome = relay.remove_member(record, state, device_id=PEER, by=SELF, root=root)
    assert outcome.epoch == 2
    assert outcome.removed == [PEER]
    peer_row = record.member(PEER)
    assert peer_row is not None
    assert peer_row.active is False
    assert peer_row.removed_at is not None
    assert record.is_burned(PEER)
    assert record.rotations["2"] == SELF
    # The secret changed and the old one is retained for reconcile, and nothing more.
    reloaded = store.load_secrets(NETWORK, root)
    assert reloaded.epoch == 2
    assert reloaded.secret != wire.b64u(b"s" * 32)
    assert reloaded.previous_epoch == 1


def test_a_rotation_inside_the_lock_is_refused() -> None:
    record, state = _record()
    relay.rotate_epoch(record, state, by=SELF, reason="member_removed", persist=False)
    with pytest.raises(types.MeshRefusal) as excinfo:
        relay.rotate_epoch(record, state, by=SELF, reason="panic", persist=False)
    assert excinfo.value.code == "rotation_in_progress"


def test_the_lowest_id_active_admin_rotates_after_a_leave() -> None:
    """Deterministic, so three peers receiving one leave do not produce three epochs
    and no election round trip is needed.

    Three steps, each of which the rule could get wrong on its own: ROLE qualifies
    (a read member is never the rotator), and among the admins the LOWEST ID wins.
    """
    record, _state = _record()
    relay.admit(
        record,
        device_id="d_" + "c" * 32,
        public_key=wire.b64u(b"c" * 32),
        role="admin",
        persist=False,
    )
    relay.admit(
        record,
        device_id=PEER,
        public_key=wire.b64u(b"b" * 32),
        role="read",
        persist=False,
    )
    # SELF is the only… no: SELF and d_c are admins; d_b is not, despite sorting
    # between them — role first, then id.
    assert relay.lowest_id_admin(record) == SELF
    self_row = record.member(SELF)
    assert self_row is not None
    self_row.role = "read"
    self_row.capabilities = sorted(types.capabilities_for_role("read"))
    assert relay.lowest_id_admin(record) == "d_" + "c" * 32
    # Promote the read member and the lower id wins again, ahead of d_c.
    peer_row = record.member(PEER)
    assert peer_row is not None
    peer_row.role = "admin"
    peer_row.capabilities = sorted(types.capabilities_for_role("admin"))
    assert relay.lowest_id_admin(record) == PEER
    relay.leave(record, device_id=PEER, persist=False)
    left_row = record.member(PEER)
    assert left_row is not None
    assert left_row.active is False
    assert relay.lowest_id_admin(record) == "d_" + "c" * 32


def test_leaving_marks_the_row_and_burns_the_id() -> None:
    record, _state = _record()
    _admit_peer(record)
    relay.leave(record, device_id=PEER, persist=False)
    assert record.is_burned(PEER)
    removed_row = record.member(PEER)
    assert removed_row is not None
    assert removed_row.removed_by == PEER


# ---------------------------------------------------------------------------
# The epoch frame, and the convergence rule
# ---------------------------------------------------------------------------


def test_the_epoch_frame_omits_the_secret_for_a_removed_recipient() -> None:
    """CONVERGENCE RULE: the frame itself must not carry the secret to a device the
    same rotation removed — a rotation that leaked its new key would be theatre."""
    record, state = _record()
    _admit_peer(record)
    relay.remove_member(record, state, device_id=PEER, by=SELF, persist=False)
    for_removed = relay.epoch_frame(record, state, reason="member_removed", target_device_id=PEER)
    assert "secret" not in for_removed
    assert for_removed["removed"] == [PEER]
    # A member that is still a member gets it.
    other = "d_" + "e" * 32
    relay.admit(
        record,
        device_id=other,
        public_key=wire.b64u(b"e" * 32),
        role="read",
        capabilities=sorted(types.capabilities_for_role("read")),
        persist=False,
    )
    for_member = relay.epoch_frame(record, state, reason="member_removed", target_device_id=other)
    assert for_member["secret"] == state.secret
    # A BROADCAST (no target) keeps the secret: that is the panic path.
    broadcast = relay.epoch_frame(record, state, reason="operator_panic")
    assert broadcast["secret"] == state.secret


def test_the_member_digest_moves_only_with_membership() -> None:
    """A digest that moved on every heartbeat would make "did your list match mine"
    unanswerable, so liveness fields are excluded from it."""
    record, _state = _record()
    digest = relay.members_digest_of(record)
    self_row = record.member(SELF)
    assert self_row is not None
    self_row.last_seen_at = 12345.0
    self_row.duplicate_count = 4
    assert relay.members_digest_of(record) == digest
    _admit_peer(record)
    assert relay.members_digest_of(record) != digest


# ---------------------------------------------------------------------------
# Applying a received rotation
# ---------------------------------------------------------------------------


def _epoch_frame(
    record: types.NetworkRecord, state: types.SecretState, *, epoch: int
) -> dict[str, Any]:
    frame = relay.epoch_frame(record, state, reason="member_removed", target_device_id=SELF)
    frame["epoch"] = epoch
    frame["rotation_id"] = PEER
    return frame


def test_a_rotation_from_an_active_member_is_applied(root: Path) -> None:
    record, state = _record()
    _admit_peer(record)
    incoming = _epoch_frame(record, state, epoch=2)
    incoming["secret"] = wire.b64u(b"n" * 32)
    outcome = relay.apply_epoch(record, state, incoming, sender_device_id=PEER, root=root)
    assert outcome.applied is True
    assert record.epoch == 2
    assert state.secret == wire.b64u(b"n" * 32)
    assert state.previous_secret == wire.b64u(b"s" * 32)
    assert record.rotations["2"] == PEER
    assert store.load(NETWORK, root).epoch == 2


def test_an_old_epoch_is_absorbed() -> None:
    """Duplicate deliveries and the ordinary race where both sides announce the same
    rotation must be idempotent, not an error."""
    record, state = _record()
    _admit_peer(record)
    frame = _epoch_frame(record, state, epoch=1)
    outcome = relay.apply_epoch(record, state, frame, sender_device_id=PEER, persist=False)
    assert outcome.detail == "already_at_epoch"
    assert record.epoch == 1


def test_a_rotation_attributed_to_another_device_is_refused() -> None:
    record, state = _record()
    _admit_peer(record)
    frame = _epoch_frame(record, state, epoch=2)
    frame["rotation_id"] = SELF
    outcome = relay.apply_epoch(record, state, frame, sender_device_id=PEER, persist=False)
    assert outcome.detail == "rotation_id_mismatch"
    assert record.epoch == 1


def test_a_member_list_that_drops_self_is_refused() -> None:
    record, state = _record()
    _admit_peer(record)
    frame = _epoch_frame(record, state, epoch=2)
    peer_row = record.member(PEER)
    assert peer_row is not None
    frame["members"] = [peer_row.to_json()]
    frame["members_digest"] = ""
    outcome = relay.apply_epoch(record, state, frame, sender_device_id=PEER, persist=False)
    assert outcome.detail == "self_absent_from_members"


def test_the_rotation_that_removes_this_device_never_awards_it_the_new_secret(
    root: Path,
) -> None:
    """A removed device must not learn the key that would let it keep reading — even
    though the frame carrying the removal is the one that would have told it."""
    record, state = _record()
    _admit_peer(record)
    frame = _epoch_frame(record, state, epoch=2)
    frame["removed"] = [SELF, PEER]
    frame["secret"] = wire.b64u(b"n" * 32)
    outcome = relay.apply_epoch(record, state, frame, sender_device_id=PEER, root=root)
    assert outcome.applied is False
    assert outcome.detail == "removed_by_this_rotation"
    assert state.secret == wire.b64u(b"s" * 32)  # unchanged
    assert record.stale == "refused_by_peers"
    assert record.is_burned(SELF)


def test_a_digest_that_does_not_match_the_members_is_refused() -> None:
    record, state = _record()
    _admit_peer(record)
    frame = _epoch_frame(record, state, epoch=2)
    frame["members_digest"] = "0" * 64
    outcome = relay.apply_epoch(record, state, frame, sender_device_id=PEER, persist=False)
    assert outcome.detail == "members_digest_mismatch"


# ---------------------------------------------------------------------------
# Panic and trust
# ---------------------------------------------------------------------------


def test_an_admin_panic_rotates_and_broadcasts_the_secret() -> None:
    record, state = _record()
    frame = relay.panic(record, state, by=SELF, is_admin=True, persist=False)
    assert record.epoch == 2
    assert frame["secret"] == state.secret
    assert frame["rotation_id"] == SELF
    assert frame["members"]


def test_a_non_admin_panic_raises_the_alarm_and_rotates_nothing() -> None:
    """The safe half of the same signal: receivers go untrusted, and no peer chooses
    a new secret for anybody."""
    record, state = _record()
    frame = relay.panic(record, state, by=PEER, is_admin=False, persist=False)
    assert record.epoch == 1
    assert "secret" not in frame
    assert frame["rotation_id"] == PEER


def test_a_received_panic_marks_the_network_untrusted(root: Path) -> None:
    record, state = _record()
    outcome = relay.apply_panic(record, {"epoch": 9}, sender_device_id=PEER, root=root)
    assert outcome.applied is True
    assert record.trust == "untrusted"
    assert PEER in record.untrusted_reason
    assert store.load(NETWORK, root).trust == "untrusted"


def test_trust_can_be_restored_and_is_stored(root: Path) -> None:
    record, _state = _record()
    relay.set_trust(record, trust="untrusted", reason="panic", root=root)
    relay.set_trust(record, trust="active", root=root)
    assert store.load(NETWORK, root).trust == "active"
    assert store.load(NETWORK, root).untrusted_reason == ""
    with pytest.raises(types.MeshRefusal):
        relay.set_trust(record, trust="whenever", root=root)


# ---------------------------------------------------------------------------
# Device-key rotation
# ---------------------------------------------------------------------------


def test_a_device_rotation_rewrites_the_row_in_place(root: Path) -> None:
    record, _state = _record()
    _admit_peer(record)
    lane = root / "peer"
    old = identity.mint(lane, name="peer")
    # The ROW is the identity: an id is a key fingerprint, so a row carrying one
    # device's id with another's key is the conflict the admission path refuses.
    lane_row = record.member(PEER)
    assert lane_row is not None
    lane_row.device_id = old.device_id
    lane_row.public_key = old.public_key
    new, previous = identity.rotate(lane, name="peer")
    statement = identity.rotation_statement(old, new, NETWORK)
    member = relay.apply_device_rotation(record, statement, root=root)
    assert member.device_id == new.device_id
    assert member.public_key == new.public_key
    # The old id is kept for a bounded window, so an in-flight link is not cut.
    assert old.device_id in member.previous_ids
    assert record.member(old.device_id) is member  # resolvable by the old id too
    assert previous.device_id == old.device_id


def test_a_rotation_statement_without_the_old_key_is_refused(root: Path) -> None:
    record, _state = _record()
    _admit_peer(record)
    lane = root / "peer"
    old = identity.mint(lane, name="peer")
    lane_row = record.member(PEER)
    assert lane_row is not None
    lane_row.device_id = old.device_id
    lane_row.public_key = old.public_key
    new, _previous = identity.rotate(lane)
    statement = identity.rotation_statement(old, new, NETWORK)
    statement["sig_old"] = wire.b64u(b"0" * 64)
    with pytest.raises(Exception) as excinfo:
        relay.apply_device_rotation(record, statement, persist=False)
    assert getattr(excinfo.value, "code", "") == "bad_rotation_statement"
