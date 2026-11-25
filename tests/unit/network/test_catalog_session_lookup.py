"""The one-session catalogue lookup behind ``peer_session_facts``.

WHY THIS FILE EXISTS. ``RemoteOwner.locate`` (the remote pilot's bind) asks the
local relay ``peer_session_facts`` about ONE session. That op used to answer from
``_fan_out_catalog`` — every member, every row, under the 12 s
``LISTING_PROBE_BUDGET_S`` — and on the 2026-11-25 mesh (232 rows, 11.56 s) one
overrun made a LIVE session read as unowned, so ``lop network sessions --slash``
failed with "could not start a runtime for this session". These tests pin the
targeted replacement on both halves:

* viewer (``_ctl_peer_facts`` / ``_peer_catalog_rows_for``): a resolved peer is
  asked ALONE, the fan-out is never run, and ``session_id`` rides the frame only
  on a link advertising ``wire.REMOTE_CATALOG_SESSION_ID_V1`` (an older owner
  receives today's exact frame and its rows are filtered here);
* owner (``_op_catalog``): ``session_id`` narrows the answer to that row, the
  row is identical to its row in the full listing, a malformed id is
  ``bad_frame``, and a frame without the field is answered as before.

No sockets: links and the owner's scan are faked at the relay's own seams, the
same way ``test_session_plane``'s fan-out dedupe test fakes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay, store, types, wire
from local_operator.network.types import MeshRefusal

OWNER = "d_" + "b" * 32
BYSTANDER = "d_" + "c" * 32
SESSION = "c7c74407768f"
OTHER = "0835f0d1a2b3"


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _admit_members(server: relay.RelayServer) -> types.NetworkRecord:
    """One network: this relay plus two members, so "asked only the owner" can fail."""
    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name="home-net",
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    relay.admit(
        record,
        device_id=server.identity.device_id,
        public_key=server.identity.public_key,
        name=server.identity.name,
        role="admin",
        added_by=server.identity.device_id,
        added_via="self",
        root=server.root,
        persist=False,
    )
    for device_id, name, key in ((BYSTANDER, "bystander", 7), (OWNER, "owner-node", 1)):
        relay.admit(
            record,
            device_id=device_id,
            public_key=wire.b64u(bytes((i * key) % 256 for i in range(32))),
            name=name,
            role="drive",
            endpoints=["127.0.0.1:1"],
            root=server.root,
            persist=False,
        )
    store.save(record, server.root)
    return record


class _Link:
    """A link that records every frame and answers with the owner's rows."""

    def __init__(self, capabilities: frozenset[str], rows: list[dict[str, Any]]) -> None:
        self.capabilities = capabilities
        self.rows = rows
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.frames.append(dict(frame))
        return {"op": "ack", "detail": {"sessions": [dict(row) for row in self.rows]}}


def _rows() -> list[dict[str, Any]]:
    return [
        {"session_id": OTHER, "state": "idle", "pid": 4242, "conversation_name": "Other"},
        {"session_id": SESSION, "state": "idle", "pid": 2407612, "conversation_name": "Live"},
    ]


def _wire_viewer(
    server: relay.RelayServer, monkeypatch: pytest.MonkeyPatch, link: _Link
) -> list[str]:
    dials: list[str] = []

    def _ensure(device_id: str, probe_timeout_s: float | None = None) -> tuple[Any, str]:
        dials.append(device_id)
        return link, ""

    def _no_fan_out() -> Any:
        raise AssertionError("peer_session_facts for a named peer ran the full fan-out")

    monkeypatch.setattr(server, "_ensure_link_with_reason", _ensure)
    monkeypatch.setattr(server, "_fan_out_catalog", _no_fan_out)
    return dials


def test_a_named_peer_is_asked_alone_with_the_session_id_on_a_capable_link(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(root)
    _admit_members(server)
    # A capable owner honours the field; the fake still returns BOTH rows so the
    # viewer's own filter is what keeps OTHER out of the answer.
    link = _Link(frozenset({wire.REMOTE_CATALOG_SESSION_ID_V1}), _rows())
    dials = _wire_viewer(server, monkeypatch, link)

    facts = server._ctl_peer_facts({"peer": "owner-node", "session_id": SESSION})

    assert dials == [OWNER], "a bystander member was dialled for one session's facts"
    assert len(link.frames) == 1
    frame = link.frames[0]
    assert frame["op"] == "net_catalog" and frame["locality"] == "remote"
    assert frame["session_id"] == SESSION
    assert facts["owned"] is True and facts["published"] is True and facts["pid"] == 2407612
    record = facts["record"]
    assert record["session_id"] == SESSION and record["locality"] == "remote"
    # The SAME peer block shape the fan-out stamps, so ``record`` readers see no change.
    assert set(record["peer"]) == {
        "device_id",
        "name",
        "network_id",
        "reachable",
        "age_s",
        "endpoints",
        "reason",
    }
    assert record["peer"]["device_id"] == OWNER and record["peer"]["reachable"] is True
    assert set(facts) == {"owned", "published", "pid", "record"}


def test_an_older_owner_gets_todays_frame_and_its_rows_are_filtered_here(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(root)
    _admit_members(server)
    link = _Link(frozenset({wire.REMOTE_ATTENTION_V1}), _rows())
    dials = _wire_viewer(server, monkeypatch, link)

    facts = server._ctl_peer_facts({"peer": OWNER, "session_id": SESSION})

    assert dials == [OWNER]
    assert len(link.frames) == 1
    assert "session_id" not in link.frames[0], "an older owner was sent a field it never knew"
    assert set(link.frames[0]) == {"op", "req", "locality"}
    assert facts["owned"] is True and facts["record"]["session_id"] == SESSION
    missing = server._ctl_peer_facts({"peer": OWNER, "session_id": "nobody-has-this"})
    assert missing == {"owned": False, "published": False, "pid": None, "record": None}


def test_an_unreachable_or_refusing_owner_reads_as_not_owned(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(root)
    _admit_members(server)
    monkeypatch.setattr(server, "_fan_out_catalog", lambda: pytest.fail("fan-out ran"))
    monkeypatch.setattr(
        server, "_ensure_link_with_reason", lambda device_id, probe_timeout_s=None: (None, "down")
    )
    not_owned = {"owned": False, "published": False, "pid": None, "record": None}
    assert server._ctl_peer_facts({"peer": OWNER, "session_id": SESSION}) == not_owned

    class _Refusing(_Link):
        def request(self, frame: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            self.frames.append(dict(frame))
            return {"op": "error", "message": "no"}

    refusing = _Refusing(frozenset({wire.REMOTE_CATALOG_SESSION_ID_V1}), _rows())
    monkeypatch.setattr(
        server, "_ensure_link_with_reason", lambda device_id, probe_timeout_s=None: (refusing, "")
    )
    assert server._ctl_peer_facts({"peer": OWNER, "session_id": SESSION}) == not_owned


def test_no_peer_keeps_the_mesh_wide_fan_out(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(root)
    calls: list[int] = []

    def _fan_out() -> Any:
        calls.append(1)
        row = dict(_rows()[1], locality="remote", peer={"device_id": OWNER})
        return {}, [row]

    monkeypatch.setattr(server, "_fan_out_catalog", _fan_out)
    facts = server._ctl_peer_facts({"session_id": SESSION})
    assert calls == [1] and facts["owned"] is True


class _ViewerLink:
    capabilities: frozenset[str] = frozenset()
    device_id = OWNER


def test_the_owner_answers_one_row_identical_to_its_full_listing_row(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(root)
    monkeypatch.setattr(server, "local_session_rows", _rows)
    link: Any = _ViewerLink()

    full = server._op_catalog(link, {"op": "net_catalog", "req": 1})
    one = server._op_catalog(link, {"op": "net_catalog", "req": 2, "session_id": SESSION})
    none = server._op_catalog(link, {"op": "net_catalog", "req": 3, "session_id": "nobody"})

    # No session_id: today's answer, keys and all.
    assert set(full) == {"complete", "device", "generated_at", "sessions"}
    assert [row["session_id"] for row in full["sessions"]] == [OTHER, SESSION]
    expected = next(row for row in full["sessions"] if row["session_id"] == SESSION)
    assert one["sessions"] == [expected]
    assert one["session_id"] == SESSION and one["complete"] is True
    assert one["device"] == full["device"]
    assert none["sessions"] == [] and none["session_id"] == "nobody"


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", 7, None, "x" * 300])
def test_a_malformed_session_id_is_a_bad_frame(
    root: Path, monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    server = _server(root)
    monkeypatch.setattr(server, "local_session_rows", _rows)
    link: Any = _ViewerLink()
    with pytest.raises(MeshRefusal) as caught:
        server._op_catalog(link, {"op": "net_catalog", "req": 1, "session_id": bad})
    assert caught.value.code == "bad_frame"


def test_the_lookup_capability_is_advertised_once() -> None:
    assert wire.LINK_CAPABILITIES.count(wire.REMOTE_CATALOG_SESSION_ID_V1) == 1
    assert wire.REMOTE_CATALOG_SESSION_ID_V1 not in types.GRANTABLE_CAPABILITIES
