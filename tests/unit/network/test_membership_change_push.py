"""Change pushes must replace the old convergence clock, not merely slow it."""

import copy
import threading
import time
from typing import Any

import pytest

from local_operator.network import identity, relay, store, types, wire
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_membership_convergence import (
    _join,
    _make,
    _rename_own_row,
    _row,
    _settled_pair,
)
from tests.unit.network.test_membership_convergence import devices as _devices_fixture

# Reuse the real relays and their exact lifecycle, rather than a second mesh rig.
devices = _devices_fixture


def _change_frame(record: Any) -> dict[str, Any]:
    return {
        "op": "net_member_list",
        "network_id": record.network_id,
        "sequence": record.sequence,
        "changed_members": [row.to_json() for row in record.members],
    }


def _wire_barrier(server: Any, link: Any) -> None:
    # The ping is queued after a notification on the SAME TCP link. Its ack
    # proves the reader finished the prior dispatch, without timing-based sleeps.
    assert link.request({"op": "ping", "req": server._next_relay_req()})["op"] == "ack"


def _negotiated_pair(a: Any, c: Any, network_id: str, link: Any) -> Any:
    """Wait for the real relay link and both initial table reads, not the CLI link."""

    def ready() -> bool:
        reverse = c.server._link_for(a.device_id, network_id=network_id)
        if reverse is None or not link.alive or not reverse.alive:
            return False
        if link.instance_id != c.server.instance_id or reverse.instance_id != a.server.instance_id:
            return False
        if not (a.server._member_push_enabled(link) and c.server._member_push_enabled(reverse)):
            return False
        if not (link.member_pulled_at and reverse.member_pulled_at):
            return False
        left, right = store.load(network_id, a.root), store.load(network_id, c.root)
        return all(
            relay.member_sync_digest(left, sender_device_id=sender)
            == relay.member_sync_digest(right, sender_device_id=sender)
            for sender in (a.device_id, c.device_id)
        )

    assert net_fixtures.wait_for(ready), "bilateral negotiated relay/table state did not settle"
    return c.server._link_for(a.device_id, network_id=network_id)


def test_two_network_pushes_stay_bound_to_the_authenticated_network(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_b, link = _settled_pair(devices, monkeypatch)
    reverse = c.server._link_for(a.device_id)
    assert reverse is not None and reverse.network_id == network_b
    record_a = copy.deepcopy(store.load(network_b, c.root))
    record_a.network_id = store.new_network_id()
    record_a.name = "other-network"
    outsider = identity.mint(c.root / "a-only-identity", name="a-only")
    unique = copy.deepcopy(record_a.member(c.device_id))
    assert unique is not None
    unique.device_id = outsider.device_id
    unique.public_key = outsider.public_key
    unique.name = "a-only"
    record_a.members.append(unique)
    store.save(record_a, c.root)

    def table(record: Any) -> list[dict[str, Any]]:
        # Wire activity may legitimately update reachability metadata. Every
        # membership/authority field, including grants and keys, must stay equal.
        volatile = {
            "last_seen_at",
            "observed_endpoint",
            "last_seen_instance",
            "duplicate_count",
            "suspect",
        }
        return [
            {key: value for key, value in row.to_json().items() if key not in volatile}
            for row in record.members
        ]

    before = table(store.load(network_b, a.root))
    c.server._publish_member_changes()
    _wire_barrier(c.server, reverse)
    after = store.load(network_b, a.root)
    print(
        "TWO_NETWORK_PUBLISH",
        "source",
        record_a.network_id,
        "link",
        network_b,
        "a_only_active_in_b",
        bool(after.member(unique.device_id)),
    )
    assert after.member(unique.device_id) is None
    assert table(after) == before
    # Defense in depth: deliberately send a misbound table over the real B link,
    # even though the publisher must now refuse to select that link for A.
    frame = _change_frame(record_a)
    frame["req"] = c.server._next_relay_req()
    reply = reverse.request(frame)
    print("TWO_NETWORK_RECEIVE", reply["op"], reply.get("code"))
    assert reply["op"] == "error"
    after = store.load(network_b, a.root)
    assert after.member(unique.device_id) is None
    assert table(after) == before


def test_an_old_push_cannot_revert_a_newer_completed_pull(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    _rename_own_row(c, network_id, "old-snapshot")
    old = store.load(network_id, c.root)
    with store.mutate(network_id, c.root) as record:
        own = record.member(c.device_id)
        assert own is not None
        own.name = "new-pull"
        own.endpoints = ["127.0.0.1:61001"]
        store.save(record, c.root)
    assert a.server._pull_members(link) == ""
    stamp = link.member_pulled_at
    pulled = _row(a, c.device_id)
    assert pulled is not None and pulled.name == "new-pull"
    reverse = c.server._link_for(a.device_id)
    assert reverse.send(_change_frame(old))
    _wire_barrier(c.server, reverse)
    row = _row(a, c.device_id)
    assert row is not None
    print(
        "STALE_PUSH_AFTER_NEW_PULL",
        row.name,
        row.endpoints,
        "pull_stamp_unchanged",
        link.member_pulled_at == stamp,
    )
    assert row.name == "new-pull"
    assert row.endpoints == ["127.0.0.1:61001"]
    assert link.member_pulled_at == stamp


def test_push_converges_without_a_pull_and_does_not_forge_answer_age(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    assert net_fixtures.wait_for(lambda: link.member_pulled_at > 0)
    # A disabled pull is the negative control: neither a timer nor the listing
    # below can rescue a missing push, and receiving it must not invent an answer.
    monkeypatch.setattr(a.server, "_pull_members", lambda link: "no_answer")
    stamp = link.member_pulled_at
    _rename_own_row(c, network_id, "changed-by-push")
    c.server._publish_member_changes()
    assert net_fixtures.wait_for(
        lambda: (row := _row(a, c.device_id)) is not None and row.name == "changed-by-push"
    )
    assert link.member_pulled_at == stamp
    report = a.server.refresh_membership()[network_id]
    assert not report.answered
    assert report.silent == [{"device_id": c.device_id, "reason": "no_table:no_answer"}]


def test_only_negotiated_change_push_links_get_the_long_ceiling(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    reverse = _negotiated_pair(a, c, network_id, link)
    # Observe autonomous settlement after both initial reads. Two manual pulls
    # race with the other endpoint's initial snapshot on a starved runner.
    assert net_fixtures.wait_for(
        lambda: link.member_pull_interval_s == relay.MEMBERSHIP_PUSH_PULL_INTERVAL_S
        and not link.member_push_unverified
    ), {"interval": link.member_pull_interval_s, "unverified": link.member_push_unverified}
    assert link.member_pull_interval_s == relay.MEMBERSHIP_PUSH_PULL_INTERVAL_S
    # Freeze ONLY subsequent verification: it must not conceal a no-op push
    # incorrectly undoing the confirmed settlement we just observed.
    monkeypatch.setattr(a.server, "refresh_membership", lambda **kwargs: {})
    stamp = link.member_pulled_at
    # The record sequence can move for liveness/metadata without changing a
    # single mergeable row. Force that snapshot to arrive AFTER the equal pull.
    with store.mutate(network_id, c.root) as record:
        store.save(record, c.root)
    assert reverse.send(_change_frame(record))
    _wire_barrier(c.server, reverse)
    assert link.member_pulled_at == stamp
    print("POST_SETTLEMENT_NOOP_PUSH", link.member_pull_interval_s, link.member_push_unverified)
    assert link.member_pull_interval_s == relay.MEMBERSHIP_PUSH_PULL_INTERVAL_S
    assert not link.member_push_unverified
    link.capabilities = frozenset(link.capabilities - {wire.MEMBER_LIST_PUSH_V1})
    link.member_pull_interval_s = relay.MEMBERSHIP_PULL_MIN_INTERVAL_S
    assert a.server._pull_members(link) == ""
    assert link.member_pull_interval_s == relay.MEMBERSHIP_PULL_MAX_INTERVAL_S
    # An old build receives neither a new opcode nor optional push fields.
    reverse = c.server._link_for(a.device_id)
    assert reverse is not None
    reverse.capabilities = frozenset(reverse.capabilities - {wire.MEMBER_LIST_PUSH_V1})
    frames = []
    monkeypatch.setattr(reverse, "send", lambda frame, **kwargs: frames.append(frame) or True)
    _rename_own_row(c, network_id, "old-peer")
    c.server._publish_member_changes()
    assert not frames


def test_unchanged_table_does_not_push_again(devices: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    reverse = _negotiated_pair(a, c, network_id, link)
    lock = threading.Lock()
    frames: list[dict[str, Any]] = []
    cycles = 0
    send = relay.PeerLink.send
    publish = c.server._publish_member_changes

    def record_send(self: relay.PeerLink, frame: dict[str, Any], **kwargs: Any) -> bool:
        accepted = send(self, frame, **kwargs)
        if accepted and self is reverse and "changed_members" in frame:
            with lock:
                frames.append(frame)
        return accepted

    def completed_cycle() -> None:
        nonlocal cycles
        publish()
        with lock:
            cycles += 1

    def snapshot() -> tuple[int, int]:
        with lock:
            return cycles, len(frames)

    def settled_after(previous: int) -> bool:
        completed, _ = snapshot()
        record = store.load(network_id, c.root)
        return completed > previous and reverse.member_push_digest == relay.member_sync_digest(
            record, sender_device_id=c.device_id
        )

    monkeypatch.setattr(relay.PeerLink, "send", record_send)
    monkeypatch.setattr(c.server, "_publish_member_changes", completed_cycle)
    # Let the ONE real publisher finish initial synchronization. Invoking it
    # manually beside its background loop creates a second, artificial producer.
    assert net_fixtures.wait_for(lambda: settled_after(0)), snapshot()
    cycle, count = snapshot()
    assert net_fixtures.wait_for(lambda: settled_after(cycle)), snapshot()
    assert snapshot()[1] == count, "an unchanged complete cycle emitted an additional push"

    cycle, count = snapshot()
    _rename_own_row(c, network_id, "new-name")
    assert net_fixtures.wait_for(
        lambda: settled_after(cycle)
        and (row := _row(a, c.device_id)) is not None
        and row.name == "new-name"
    ), snapshot()
    changed_cycle, _ = snapshot()
    # A second completed cycle proves quiescence, not merely that the first
    # expected frame arrived before a duplicate still queued on another thread.
    assert net_fixtures.wait_for(lambda: settled_after(changed_cycle)), snapshot()
    assert snapshot()[1] == count + 1
    assert frames[-1]["op"] == "net_member_list"
    assert "req" not in frames[-1]


def test_push_from_an_unnegotiated_peer_is_refused(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    link.capabilities = frozenset(link.capabilities - {wire.MEMBER_LIST_PUSH_V1})
    stamp = link.member_pulled_at
    with pytest.raises(relay.MeshRefusal):
        a.server._op_member_list(link, {"changed_members": []})
    assert link.member_pulled_at == stamp
    assert time.time() >= stamp


def test_a_change_notification_has_no_request_to_acknowledge(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    record = store.load(network_id, c.root)
    assert (
        a.server.dispatch(
            link,
            _change_frame(record),
        )
        is None
    )


def test_an_inflight_answer_cannot_revert_a_newer_push(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    request = link.request
    stamp = link.member_pulled_at

    def overtaken(frame: dict[str, Any], *, timeout: float) -> dict[str, Any] | None:
        reply = request(frame, timeout=timeout)
        _rename_own_row(c, network_id, "newer-than-answer")
        record = store.load(network_id, c.root)
        a.server._op_member_list(link, _change_frame(record))
        return reply

    monkeypatch.setattr(link, "request", overtaken)
    assert a.server._pull_members(link) == "stale_answer"
    row = _row(a, c.device_id)
    assert row is not None and row.name == "newer-than-answer"
    assert link.member_pulled_at == stamp
    assert link.member_push_unverified
    assert link.member_pull_interval_s == relay.MEMBERSHIP_PULL_MIN_INTERVAL_S


@pytest.mark.parametrize("supplier_role", ["read", "admin"])
def test_new_rows_require_a_solicited_answer_and_supplier_standing(
    devices: Any, monkeypatch: pytest.MonkeyPatch, supplier_role: str
) -> None:
    """Real encrypted pushes cannot admit; their subsequent pull is peer-bound."""
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    reverse = _negotiated_pair(a, c, network_id, link)
    pull = a.server._pull_members
    # Disable the background rescue path before publishing: the push assertion
    # must not accidentally observe an admission made by a concurrent pull.
    monkeypatch.setattr(a.server, "_pull_members", lambda link: "no_answer")
    with store.mutate(network_id, a.root) as record:
        supplier = record.member(c.device_id)
        assert supplier is not None
        supplier.role = supplier_role
        supplier.capabilities = sorted(types.capabilities_for_role(supplier_role))
        store.save(record, a.root)
    newcomer = identity.mint(c.root / "asserted-device", name="asserted")
    with store.mutate(network_id, c.root) as record:
        row = types.MemberRecord(
            device_id=newcomer.device_id,
            public_key=newcomer.public_key,
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_by=a.device_id,  # Claimed provenance cannot replace link standing.
        )
        record.members.append(row)
        store.save(record, c.root)
    assert reverse.send(_change_frame(record))
    _wire_barrier(c.server, reverse)
    assert store.load(network_id, a.root).member(newcomer.device_id) is None
    assert pull(link) == ""
    adopted = store.load(network_id, a.root).member(newcomer.device_id)
    print("BOUND_TABLE_ANSWER", supplier_role, "adopted", adopted is not None)
    if supplier_role == "admin":
        assert adopted is not None and adopted.role == "admin"
    else:
        assert adopted is None


def test_removal_and_epoch_still_reach_every_peer(
    devices: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, c, network_id, link = _settled_pair(devices, monkeypatch)
    removed = _make(devices, "removed")
    assert _join(removed, inviter=c, monkeypatch=monkeypatch)["device_id"] == removed.device_id
    # Pairing uses a short-lived CLI connection. Establish the relay's own link
    # before removal, so the removed device can receive its epoch notification.
    removed_link, reason = removed.server.dial(network_id, host=f"{c.host}:{c.port}", epoch=1)
    assert removed_link is not None, reason
    assert net_fixtures.wait_for(
        lambda: (live := c.server._link_for(removed.device_id)) is not None and live.alive
    )
    assert a.server._pull_members(link) == ""
    assert store.load(network_id, a.root).member(removed.device_id) is not None
    old_secret = store.load_secrets(network_id, removed.root).secret
    result = c.server._ctl_member_rm({"network": network_id, "device_id": removed.device_id})
    assert result["epoch"] == 2

    def all_removed() -> bool:
        records = [store.load(network_id, peer.root) for peer in (a, c, removed)]
        return all(
            record.epoch == 2
            and removed.device_id in record.removed_ids
            and (row := record.member(removed.device_id)) is not None
            and not row.active
            for record in records
        )

    assert net_fixtures.wait_for(all_removed)
    assert (
        store.load_secrets(network_id, a.root).secret
        == store.load_secrets(network_id, c.root).secret
    )
    assert store.load_secrets(network_id, removed.root).secret == old_secret
    print(
        "REMOVAL_ALL_PEERS",
        "epochs",
        [store.load(network_id, p.root).epoch for p in (a, c, removed)],
    )
