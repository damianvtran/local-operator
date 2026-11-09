"""Change pushes must replace the old convergence clock, not merely slow it."""

import time
from typing import Any

import pytest

from local_operator.network import relay, store, wire
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_membership_convergence import (
    _rename_own_row,
    _row,
    _settled_pair,
)
from tests.unit.network.test_membership_convergence import devices as _devices_fixture

# Reuse the real relays and their exact lifecycle, rather than a second mesh rig.
devices = _devices_fixture


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
            {"op": "net_member_list", "changed_members": [row.to_json() for row in record.members]},
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
        a.server._op_member_list(
            link, {"changed_members": [row.to_json() for row in record.members]}
        )
        return reply

    monkeypatch.setattr(link, "request", overtaken)
    assert a.server._pull_members(link) == "table_changed"
    assert _row(a, c.device_id).name == "newer-than-answer"
    assert link.member_pulled_at == stamp
    assert link.member_push_unverified
    assert link.member_pull_interval_s == relay.MEMBERSHIP_PULL_MIN_INTERVAL_S
