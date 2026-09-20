"""The store: atomic writes, quarantine, prune, the queue rule, run/peers."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from local_operator.network import store, types, wire

NETWORK = "n_0123456789abcdef01234567"


def _record(
    network_id: str = NETWORK, name: str = "home-net", epoch: int = 1
) -> types.NetworkRecord:
    return types.NetworkRecord(
        network_id=network_id,
        name=name,
        epoch=epoch,
        created_by="d_" + "a" * 32,
        self_device_id="d_" + "a" * 32,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )


def test_save_and_load_round_trip(root: Path) -> None:
    record = _record()
    path = store.save(record, root)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    loaded = store.load(NETWORK, root)
    assert loaded.name == record.name
    assert loaded.sequence == record.sequence == 1
    # A second save bumps the sequence, which is what epoch broadcasts carry.
    store.save(loaded, root)
    assert store.load(NETWORK, root).sequence == 2


def test_a_write_leaves_no_temporary_behind(root: Path) -> None:
    store.save(_record(), root)
    assert list(store.networks_dir(root).glob(".*")) == []


def test_secrets_round_trip_and_keep_exactly_one_generation(root: Path) -> None:
    state = types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"a" * 32))
    store.save_secrets(state, root)
    assert store.load_secrets(NETWORK, root).secret == state.secret
    state.rotate(wire.b64u(b"b" * 32), 2)
    store.save_secrets(state, root)
    reloaded = store.load_secrets(NETWORK, root)
    assert (reloaded.epoch, reloaded.secret) == (2, wire.b64u(b"b" * 32))
    assert (reloaded.previous_epoch, reloaded.previous_secret) == (1, wire.b64u(b"a" * 32))
    state.rotate(wire.b64u(b"c" * 32), 3)
    store.save_secrets(state, root)
    third = store.load_secrets(NETWORK, root)
    # A third epoch is dropped: the claim is "two, never more", and it is asserted.
    assert third.previous_epoch == 2
    assert third.previous_secret == wire.b64u(b"b" * 32)


def test_a_corrupt_record_is_quarantined_not_deleted(root: Path) -> None:
    """A membership list is the one file a mistake must not quietly reinterpret, so
    an unparsable record is SET ASIDE and reported, never dropped."""
    store.save(_record(), root)
    store.save(_record("n_ffffffffffffffffffffffff", name="other"), root)
    broken = store.record_path("n_ffffffffffffffffffffffff", root)
    broken.write_text("{not json", encoding="utf-8")
    listed = store.list_networks(root)
    assert [record.network_id for record in listed] == [NETWORK]
    assert broken.with_name(broken.name + store.CORRUPT_SUFFIX).exists()
    assert not broken.exists()


def test_a_record_with_the_wrong_shape_is_quarantined_too(root: Path) -> None:
    path = store.record_path("n_ffffffffffffffffffffffff", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"network_id": "n_x", "members": "not-a-list"}), encoding="utf-8")
    assert store.list_networks(root) == []
    assert path.with_name(path.name + store.CORRUPT_SUFFIX).exists()


def test_forget_removes_this_devices_own_copy_only(root: Path) -> None:
    record = _record()
    store.save(record, root)
    store.save_secrets(types.SecretState(network_id=NETWORK, epoch=1, secret="x"), root)
    removed = store.forget(NETWORK, root)
    assert len(removed) == 2
    assert not store.record_path(NETWORK, root).exists()
    assert not store.secrets_path(NETWORK, root).exists()


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


def test_prune_drops_old_invites_but_never_a_burned_id(root: Path) -> None:
    record = _record()
    record.invites.append(
        types.InviteRecord(invite_id="old", minted_at=0.0, ttl_s=60.0, state="consumed")
    )
    record.invites.append(types.InviteRecord(invite_id="recent", minted_at=time_now(), ttl_s=600.0))
    record.members.append(
        types.MemberRecord(
            device_id="d_" + "b" * 32,
            removed_at=0.0,
            removed_by="d_" + "a" * 32,
            lifecycle="expired",
        )
    )
    record.removed_ids.append("d_" + "b" * 32)
    outcome = store.prune(record, now=time_now())
    assert [invite.invite_id for invite in record.invites] == ["recent"]
    assert outcome["tombstone_rows_dropped"] == 1
    # THE POINT: the row goes, the id stays, so the id cannot be re-admitted.
    assert record.removed_ids == ["d_" + "b" * 32]
    assert record.is_burned("d_" + "b" * 32)


def test_prune_keeps_a_recent_tombstone_row(root: Path) -> None:
    record = _record()
    record.members.append(
        types.MemberRecord(device_id="d_" + "b" * 32, removed_at=time_now(), lifecycle="expired")
    )
    outcome = store.prune(record, now=time_now())
    assert outcome["tombstone_rows_dropped"] == 0
    assert len(record.members) == 1


def time_now() -> float:
    import time

    return time.time()


# ---------------------------------------------------------------------------
# The durable outbox, and the convergence rule
# ---------------------------------------------------------------------------


def test_a_secret_is_never_queued_for_a_removed_recipient(root: Path) -> None:
    """CONVERGENCE RULE at the writer: a file is the one place a secret outlives the
    decision to stop trusting a device."""
    frame = {"op": "net_epoch", "epoch": 5, "secret": wire.b64u(b"s" * 32)}
    with pytest.raises(types.MeshRefusal) as excinfo:
        store.enqueue_frame("d_" + "b" * 32, frame, removed=True, root=root)
    assert excinfo.value.code == "removed_recipient"
    assert store.queued_frames("d_" + "b" * 32, root) == []
    # For an ACTIVE member the same frame is queued.
    store.enqueue_frame("d_" + "b" * 32, frame, removed=False, root=root)
    queued = store.queued_frames("d_" + "b" * 32, root)
    assert len(queued) == 1
    path, stored = queued[0]
    assert stored["secret"] == frame["secret"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_queued_frames_come_back_in_write_order_and_can_be_dropped(root: Path) -> None:
    for index in range(3):
        store.enqueue_frame(
            "d_" + "b" * 32,
            {"op": "net_epoch", "epoch": index},
            removed=False,
            root=root,
            now=100.0 + index,
        )
    queued = store.queued_frames("d_" + "b" * 32, root)
    assert [frame["epoch"] for _path, frame in queued] == [0, 1, 2]
    store.drop_frame(queued[0][0])
    assert len(store.queued_frames("d_" + "b" * 32, root)) == 2


def test_a_torn_queued_frame_does_not_block_the_queue(root: Path) -> None:
    store.enqueue_frame("d_" + "b" * 32, {"op": "net_epoch", "epoch": 1}, removed=False, root=root)
    torn = store.peer_outbox_dir("d_" + "b" * 32, root) / "0000000000000-1-aaaa.frame"
    torn.write_text("{torn", encoding="utf-8")
    assert len(store.queued_frames("d_" + "b" * 32, root)) == 1
    assert not torn.exists()


# ---------------------------------------------------------------------------
# The run/peers namespace (A2)
# ---------------------------------------------------------------------------


def _peer_record() -> types.PeerRecord:
    return types.PeerRecord(
        pid=os.getpid(),
        device_id="d_" + "a" * 32,
        device_name="laptop",
        instance_id="i_1",
        control_port=5123,
        control_key=wire.b64u(b"k" * 32),
        listen={"address": "127.0.0.1", "port": 4097, "advertised": []},
        networks=[{"network_id": NETWORK, "name": "home-net", "epoch": 1}],
    )


def test_a_peer_record_is_published_under_its_own_namespace(root: Path) -> None:
    """``lop sessions`` keeps meaning sessions: a relay record lives in a FOURTH
    namespace, so a session reader cannot mistake it for a session."""
    path = store.publish_peer_record(_peer_record(), root)
    assert path.parent.name == "peers"
    assert store.PEERS_RUN_DIRNAME not in ("", "run/mobile")
    assert types.PEERS_RUN_DIRNAME == "run/peers"
    scanned = store.scan_peer_records(root)
    assert len(scanned) == 1
    record, state = scanned[0]
    assert isinstance(record, types.PeerRecord)
    assert record.networks[0]["network_id"] == NETWORK
    store.unpublish_peer_record(os.getpid(), root)
    assert store.scan_peer_records(root) == []


def test_a_peer_record_carries_no_key_material(root: Path) -> None:
    """Stated as an assertion because the natural next change to this record is
    "add something helpful", and the helpful thing is always a key."""
    from local_operator.network import identity

    mine = identity.mint(root)
    record = _peer_record()
    record.device_id = mine.device_id
    payload = json.dumps(record.to_json())
    assert mine.private_key not in payload
    assert mine.public_key not in payload
    # The one key it MAY carry is the loopback control key, whose protection is the
    # 0600 file and the account — the same boundary a session record uses.
    assert record.control_key in payload


def test_peer_record_from_json_drops_unknown_keys() -> None:
    record = types.PeerRecord.from_json({"pid": 7, "heartbeat_at": 1.0, "future_field": "x"})
    assert record.pid == 7
    assert not hasattr(record, "future_field")


# ---------------------------------------------------------------------------
# Resolution by id, name, or not at all
# ---------------------------------------------------------------------------


def test_match_networks_prefers_the_id_then_the_name() -> None:
    first = _record("n_" + "1" * 24, name="home")
    second = _record("n_" + "2" * 24, name="home")
    third = _record("n_" + "3" * 24, name="lab")
    records = [first, second, third]
    assert store.match_networks(records, first.network_id) == [first]
    # Names are NOT unique by design, so an ambiguous name returns both and the
    # caller refuses rather than picking one.
    assert store.match_networks(records, "home") == [first, second]
    assert store.match_networks(records, "lab") == [third]
    assert store.match_networks(records, "missing") == []
