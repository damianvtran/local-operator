"""The store: atomic writes, quarantine, prune, the queue rule, run/peers."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path
from typing import Any

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


def test_two_threads_writing_one_record_lose_nothing_and_reverse_nothing(root: Path) -> None:
    """QA round 15, Q15-2: this package serialised READERS and no writers.

    The staging name was ``.{name}.{pid}.tmp``, and ``os.getpid()`` is not unique
    inside a process, so every thread of one process writing a record shared one
    staging file — one thread's ``os.replace`` or its ``finally`` unlink took the
    file away under another's ``os.chmod`` (``FileNotFoundError``, measured on
    ``test_session_plane``), and the body that landed was whichever thread last
    wrote into the shared file rather than whichever last called ``save``.

    The other half of the same fault is the sequence. ``save`` bumped the copy its
    CALLER held, so two writers that had read the same file both stamped the same
    next number: one update vanished with no trace, because nothing on the read
    path can tell a stale write from a fresh one. The two halves are one fault
    because ONE window produces both — the writers share the staging file, and
    neither the bytes nor the counter is serialised.

    SO THE CELL PINS BOTH, deterministically rather than by racing:

    * (a) every write lands, and (b) nothing runs backwards: eight writers must
      stamp eight distinct consecutive numbers, 2..9 from a file primed at 1. A
      pre-fix writer stamps the number its own stale copy implied, so the run
      shows repeats (the lost update) instead;
    * no write may die. The shared staging file took the file away under another
      thread's ``os.chmod``, so writers raise ``FileNotFoundError`` here —
      collected rather than escaping a thread, because an exception raised in a
      pool is how the traceback that motivated this cell got lost.

    The barrier puts every writer on the same stale copy — not a widened window
    but the exact state a relay thread and the CLI hold.
    """
    writers = 8
    # Prime the file at sequence 1: every writer's copy is loaded from it, so the
    # stamps below must be 2..9 rather than 1..8.
    store.save(_record(), root)
    barrier = threading.Barrier(writers, timeout=30)
    guard = threading.Lock()
    stamped: list[int] = []
    errors: list[BaseException] = []

    def write_one() -> None:
        try:
            record = store.load(NETWORK, root)
            barrier.wait()
            store.save(record, root)
            # ``save`` stamps the sequence ON the caller's record, which is what
            # the relay carries into an epoch broadcast (`relay.py`, net_epoch).
            with guard:
                stamped.append(record.sequence)
        except BaseException as exc:  # noqa: BLE001 - the assertion is the report
            with guard:
                errors.append(exc)

    threads = [
        threading.Thread(target=write_one, name=f"mesh-store-writer-{index}")
        for index in range(writers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a writer never returned"
    # ONE assertion for both halves, so a pre-fix run reports the whole fault
    # rather than whichever half it reached first: no write may die, and the
    # writers that did land must have stamped 2..9 with no repeat.
    assert (errors, sorted(stamped)) == ([], list(range(2, writers + 2))), (
        "eight threads writing one record must all land, each on the file it "
        f"replaced: {len(errors)} raised "
        f"{sorted({type(exc).__name__ for exc in errors})}, and those that landed stamped "
        f"{sorted(stamped)} rather than {list(range(2, writers + 2))}"
    )
    assert store.load(NETWORK, root).sequence == writers + 1


def test_mutate_reads_after_the_lock_and_cannot_ride_an_earlier_write(root: Path) -> None:
    """``mutate`` is the read-modify-write the store's lock alone could not give.

    The lock serialises writers, and a caller's own ``load`` → mutate → ``save`` is
    a LONGER span than one write: measured on the pair listener, the record is read,
    a human is asked to compare a code, and the admission is written back minutes
    later from that copy — reverting whatever the heartbeat, a membership pull or a
    peer's rotation wrote meanwhile, with a HIGHER sequence on the reverted file.

    So the cell pins the property rather than a race: a body's read happens AFTER
    the lock is taken, which is what makes a second read-modify-write of the same
    record see the first one's write. Nested here in one thread — the same sequence
    two relay threads get, without the timing that makes a racing cell flaky — and
    the two appends must BOTH be on disk at the end. Pre-fix there is nothing to
    call: the shape in use was ``load`` beside ``save``, and the second read would
    hand back the state the first write replaced.
    """
    store.save(_record(), root)
    with store.mutate(NETWORK, root) as first:
        first.removed_ids.append("d_first")
        store.save(first, root)
        with store.mutate(NETWORK, root) as second:
            assert second.removed_ids == ["d_first"], "a read inside the lock sees the write"
            second.removed_ids.append("d_second")
            store.save(second, root)
    final = store.load(NETWORK, root)
    assert final.removed_ids == ["d_first", "d_second"]


def test_threads_that_read_modify_write_through_mutate_lose_nothing(root: Path) -> None:
    """Six threads appending one row each: all six must be on disk at the end.

    The sibling cell above races ``save`` and pins the sequence; this one races the
    whole read-modify-write, which is the span a relay thread actually holds. The
    barrier puts every writer at its own read at the same instant, and what makes the
    cell deterministic is that ``mutate`` refuses to let two of them overlap: the
    second is still waiting for the lock when the first writes, so it reads the
    first's row rather than the file the first replaced.

    Passing this cell does NOT prove the record cannot be written concurrently —
    only that the writers that go through ``mutate`` do not lose each other's edits.
    """
    store.save(_record(), root)
    writers = 6
    barrier = threading.Barrier(writers, timeout=30)
    guard = threading.Lock()
    errors: list[BaseException] = []

    def append_one(index: int) -> None:
        try:
            barrier.wait()
            with store.mutate(NETWORK, root) as record:
                record.removed_ids.append(f"d_{index}")
                store.save(record, root)
        except BaseException as exc:  # noqa: BLE001 - the assertion is the report
            with guard:
                errors.append(exc)

    threads = [
        threading.Thread(target=append_one, args=(index,), name=f"mesh-rmw-{index}")
        for index in range(writers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a writer never returned"
    assert errors == [], f"{len(errors)} writer(s) raised: {errors[:2]}"
    assert sorted(store.load(NETWORK, root).removed_ids) == [
        f"d_{index}" for index in range(writers)
    ]


def test_one_lock_per_target_is_released_and_not_leaked(root: Path) -> None:
    """The registry of write locks must not grow with the traffic.

    The outbox queues write one file PER FRAME, so an entry kept per target ever
    written would be a leak proportional to the messages sent — which is why the
    entry is dropped as its last user leaves rather than kept for the process.
    """
    assert store._WRITE_LOCKS == {}
    for index in range(20):
        store.enqueue_frame(
            "d_" + "b" * 32, {"op": "net_epoch", "epoch": index}, removed=False, root=root
        )
    assert store._WRITE_LOCKS == {}, "every lock must go when its last writer has"


def test_an_invite_token_arrives_by_rename_and_is_never_partial(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token is a BEARER CREDENTIAL, and it used to be written in place.

    ``path.write_text`` created it at the umask's mode and chmodded afterwards, so
    for the length of the write the token was readable to other local users and a
    reader could observe a truncated one. It now stages beside its target and
    renames in, which is pinned here by the rename itself rather than by the final
    mode: the mode is 0600 either way, and the window is the whole point.
    """
    renames: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        renames.append((os.fspath(source), os.fspath(destination)))
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    path = store.save_invite_token("inv_round_trip", "tok-abc", root)
    assert path.read_text(encoding="utf-8") == "tok-abc\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [destination for _source, destination in renames] == [
        os.fspath(path)
    ], "the token must be renamed into place, never written at the target path"
    source = renames[0][0]
    assert os.path.dirname(source) == os.path.dirname(
        os.fspath(path)
    ), "staging beside the target is what makes the rename atomic"
    assert list(store.outbox_dir(root).glob(".*")) == []


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
