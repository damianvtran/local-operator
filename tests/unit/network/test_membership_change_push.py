"""Change pushes must replace the old convergence clock, not merely slow it."""

import copy
import time
from typing import Any

import pytest

from local_operator.network import identity, relay, store, wire
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_membership_convergence import (
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
    assert a.server._pull_members(link) == ""
    assert a.server._pull_members(link) == ""
    assert link.member_pull_interval_s == relay.MEMBERSHIP_PUSH_PULL_INTERVAL_S
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
    reverse = c.server._link_for(a.device_id)
    assert reverse is not None
    frames = []
    monkeypatch.setattr(reverse, "send", lambda frame, **kwargs: frames.append(frame) or True)
    c.server._publish_member_changes()
    count = len(frames)
    c.server._publish_member_changes()
    assert len(frames) == count
    _rename_own_row(c, network_id, "new-name")
    c.server._publish_member_changes()
    assert len(frames) == count + 1
    assert frames[-1]["op"] == "net_member_list"
    assert "changed_members" in frames[-1]
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
