"""Remote attention over two REAL relays: the owner stamps only its origin's rows.

The strongest proof this repo can give without a second machine: two
``RelayServer``s in one process, paired by the shared fixture's real ceremony,
with a real authenticated ``PeerLink`` between them and the production
``net_catalog`` op answering over it. Device B dials device A, so A is the OWNER
answering and B the VIEWER asking; the session lives on A's disk with a stamp
naming its origin.

Pinned here, each over the wire rather than against a helper:

* the origin viewer gets ``attention``/``notices`` on the session's row, and the
  origin's watcher turns that reply into exactly one ``remote-done:`` row and
  one ``remote-notice:`` row in ITS inbox;
* a viewer that is NOT the session's origin gets the row without either field;
* an OLD viewer (one that does not advertise ``remote-attention-v1``; the list
  is narrowed exactly as ``test_relay_e2e``'s old-peer cells do) gets the exact
  row shape it always did.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay, types, wire
from local_operator.network.projection import PeerRow
from local_operator.notifications import remote_attention as ra
from local_operator.notifications.store import NotificationStore
from local_operator.session import peer_rows
from local_operator.session.attention import AttentionStore
from local_operator.session.placement import MeshStamp, SessionPlacement, write_stamp
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _live_link,
    devices,
)

SID = "c0ffee123456"
# A real UUID: the completion store refuses any other token shape.
TOKEN = str(uuid.uuid5(uuid.NAMESPACE_URL, "remote-attention-relay-test"))

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture under a name these tests can take (F811)."""
    value: Devices = request.getfixturevalue("devices")
    return value


def _own_session(owner: relay.RelayServer, origin_device: str) -> None:
    """A session on ``owner``'s disk that ``origin_device`` started there."""
    directory = owner.root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    write_stamp(
        owner.root,
        MeshStamp(
            session_id=SID,
            network_id="n_test",
            home_device=owner.identity.device_id,
            placement=SessionPlacement(
                mode="peer",
                network_id="n_test",
                home_device=owner.identity.device_id,
                stamp_revision=1,
            ),
            origin={"kind": "user", "source_device": origin_device, "source_session_id": ""},
        ),
    )
    AttentionStore(owner.root / "attention.db").publish(
        f"session/{SID}", TOKEN, "anchor-1", "complete"
    )
    NotificationStore(owner.root).notify(
        SID, title="Release cut — sign-off needed", priority="high", key="release"
    )


def _catalog_row(link: relay.PeerLink) -> dict[str, Any]:
    reply = link.request({"op": "net_catalog", "req": 41, "locality": "remote"})
    assert reply is not None and reply.get("op") == "ack", reply
    rows = [row for row in reply["detail"]["sessions"] if row.get("session_id") == SID]
    assert len(rows) == 1, reply["detail"]["sessions"]
    return dict(rows[0])


def test_the_origin_gets_both_fields_and_its_watcher_raises_the_rows(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = pair
    _record, link = _live_link(pair, monkeypatch)
    try:
        _own_session(server_a, server_b.identity.device_id)
        assert wire.REMOTE_ATTENTION_V1 in link.capabilities
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert row["attention"] == {"token": TOKEN, "outcome": "complete"}
    [notice_block] = row["notices"]
    assert notice_block["key"] == "release"
    assert notice_block["priority"] == "high"

    # THE ORIGIN HALF, from the bytes that crossed: the production parse, the
    # production row builder, the production watcher.
    owner = server_a.identity.device_id

    class _Peer:
        device_id = owner
        name = "lop-remote-node-2"
        network_id = "n_test"
        reachable = True
        reason = ""

    class _Catalog:
        def peers(self) -> list[Any]:
            return [_Peer()]

        def rows(self) -> list[Any]:
            return [PeerRow.from_json(row, device_id=owner)]

    monkeypatch.setattr(ra, "_deliver_async", lambda *_a, **_k: None)
    monkeypatch.setattr(ra, "_deliver_done_banner", lambda *_a, **_k: None)
    head = tmp_path / "head-inbox"
    rows = peer_rows.peer_session_rows(None, ttl_s=0, catalog=_Catalog())
    ra.RemoteAttentionWatcher(head).observe(rows)
    items, _ = NotificationStore(head).list_items(state="all", limit=10)
    assert sorted(item["key"] for item in items) == sorted(
        [ra.done_key(owner, SID), ra.notice_key(owner, SID, "release")]
    )
    notice = next(item for item in items if item["key"].startswith("remote-notice:"))
    assert notice["priority"] == "high"


def test_a_non_origin_viewer_gets_neither_field(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _server_b, _host, _port = pair
    _record, link = _live_link(pair, monkeypatch)
    try:
        # Started by SOME OTHER device: B is a co-member, not the operator's origin.
        _own_session(server_a, "d_someone_else")
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert "attention" not in row and "notices" not in row


def test_an_old_viewer_gets_the_row_shape_it_always_did(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(c for c in wire.LINK_CAPABILITIES if c != wire.REMOTE_ATTENTION_V1),
    )
    server_a, server_b, _host, _port = pair
    _record, link = _live_link(pair, monkeypatch)
    try:
        _own_session(server_a, server_b.identity.device_id)
        assert wire.REMOTE_ATTENTION_V1 not in link.capabilities
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert "attention" not in row and "notices" not in row


def test_types_module_is_untouched_by_the_capability() -> None:
    """The capability is a LINK advertisement, never a role grant."""
    assert wire.REMOTE_ATTENTION_V1 in wire.LINK_CAPABILITIES
    assert wire.REMOTE_ATTENTION_V1 not in types.GRANTABLE_CAPABILITIES
