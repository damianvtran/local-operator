"""The complete peer-source contract: negotiation, typed purposes, the lane.

The stored-read TRANSPORT slice is pinned in ``test_peer_journal_order.py``;
this file pins the COMPLETE contract on top of it (docs/DESKTOP_API.md
"Complete peer-source contract"):

* the vocabulary and bound ordering the negotiation rests on;
* the owner's typed source purposes — metadata / payload — with their
  partitions, ACK and mismatch corrections, on the REAL relay pair;
* rejections on BOTH trust boundaries (local control and authenticated owner),
  never silent truncation;
* the deferred availability lane: coalescing, the pending cap, once-answer,
  rechecks after the wait, and the stopped-owner/no-lease property;
* the bridge's wake machinery: a metadata-only ``journal_order_ready`` for
  complete-declared subscribers only, the reconcile fallback, and the
  no-ACK-means-unsupported rule.

Bodies are never printed: counts and ids only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import projection, relay, wire
from local_operator.network.types import LinkContext, Refusal
from local_operator.session import transcript as tr
from local_operator.session import transcript_index as ti
from local_operator.wakes import store as wake_store
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixture registration
    devices,
)
from tests.unit.network.test_session_plane import (
    SESSION,
    Devices,
    _call,
    _dial_to,
    _listen,
    _pair,
    _seed_journal,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    return request.getfixturevalue("devices")


def _ready(root: Path, session_id: str = SESSION) -> Any:
    async def build() -> Any:
        task, _ = ti.start_refresh(root, session_id)
        return await task

    index = asyncio.run(build())
    assert ti.fresh_resident(root, session_id) is not None
    return index


def _write_session(root: Path, session_id: str, entries: list[tr.TranscriptEntry]) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / tr.TRANSCRIPT_FILENAME).write_text(
        "".join(entry.to_json() + "\n" for entry in entries)
    )
    return directory


def _row(
    id_: str, seq: int, role: str = "user", text: str = "fixture", **extra: Any
) -> tr.TranscriptEntry:
    return tr.TranscriptEntry(
        id_,
        float(seq),
        tr.ENTRY_MESSAGE,
        {"kind": "message", "role": role, "content": [{"text": text}], **extra},
    )


def _read(server: relay.RelayServer, session_id: str = SESSION, **fields: Any) -> dict[str, Any]:
    return server._op_session_history(
        relay.PeerLink.__new__(relay.PeerLink),
        {"session_id": session_id, "limit": 2, "req": 1, **fields},
    )


# ---------------------------------------------------------------------------
# Vocabulary and bounds
# ---------------------------------------------------------------------------


def test_vocabulary_and_bound_ordering() -> None:
    """The negotiation's own facts: one string, one flag, one bound ordering."""
    assert wire.PEER_JOURNAL_SOURCE_V1 == "peer-journal-source-v1"
    assert wire.PEER_JOURNAL_SOURCE_V1 in wire.LINK_CAPABILITIES
    assert relay.SOURCE_AVAILABILITY_WAIT_S < relay.session_history_hop_bound_s()
    assert relay.SOURCE_AVAILABILITY_MAX_PENDING >= 1
    from local_operator.session.runtime.types import (
        PEER_JOURNAL_SOURCE_CAPABILITY,
        PEER_JOURNAL_SOURCE_FLAG,
    )

    assert PEER_JOURNAL_SOURCE_CAPABILITY == wire.PEER_JOURNAL_SOURCE_V1
    assert PEER_JOURNAL_SOURCE_FLAG == "peer_journal_source"
    from local_operator.server.features import feature_flags

    assert feature_flags().get("peer_journal_source") == 1


# ---------------------------------------------------------------------------
# Owner purposes: metadata
# ---------------------------------------------------------------------------


def test_metadata_purpose_ready_facts_and_ack(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    ids = _seed_journal(b.root, SESSION, ["", "", ""])
    index = _ready(b.root)
    detail = _read(
        b,
        journal_order=True,
        order_ids=[ids[0], "absent"],
        peer_journal_source=True,
        source_read="metadata",
    )
    assert detail["entries"] == []
    assert detail["journal_order_state"] == "ready"
    assert detail["journal_generation"] == index.journal_generation
    assert detail["journal_ordinals"] == {ids[0]: index.journal_rows[ids[0]]}
    assert detail["journal_missing_ids"] == ["absent"]
    assert detail["peer_journal_source"] is True
    assert "reconcile" not in detail
    # Facts only: no cursor/window/frontend/body effects on a metadata read.
    assert detail["has_more"] is False and detail["cursor_missing"] is False


def test_metadata_building_is_honest_and_never_fabricates_ordinals(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    _a, b, _, _ = peer_pair
    ids = _seed_journal(b.root, SESSION, [""])
    # Hold the refresh scheduling edge still: the cold answer must be honest.
    monkeypatch.setattr(ti, "start_refresh", lambda *args, **kwargs: None)
    detail = _read(
        b, journal_order=True, order_ids=[ids[0]], peer_journal_source=True, source_read="metadata"
    )
    assert detail["journal_order_state"] == "building"
    assert detail["peer_journal_source"] is True
    assert "journal_generation" not in detail
    assert "journal_ordinals" not in detail and "journal_missing_ids" not in detail


def test_metadata_expected_generation_mismatch_reconciles(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    ids = _seed_journal(b.root, SESSION, [""])
    index = _ready(b.root)
    detail = _read(
        b,
        journal_order=True,
        order_ids=[ids[0]],
        peer_journal_source=True,
        source_read="metadata",
        expected_journal_generation="not-the-one",
    )
    assert detail["journal_order_state"] == "unavailable"
    assert detail["reconcile"] is True
    assert detail["peer_journal_source"] is True
    # The CURRENT generation rides the answer so a re-read is owed; the answer
    # is not an authority to install it (no rows, no ordinals beside it).
    assert detail["journal_generation"] == index.journal_generation
    assert "journal_ordinals" not in detail


# ---------------------------------------------------------------------------
# Owner purposes: payload
# ---------------------------------------------------------------------------


def test_payload_purpose_serves_bodies_partitions_and_dependencies(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    owner = _row(
        "call-owner",
        1,
        role="assistant",
        tool_calls=[{"id": "c1", "name": "read", "arguments": {}}],
    )
    target = tr.TranscriptEntry(
        "target",
        2.0,
        tr.ENTRY_MESSAGE,
        {
            "kind": "message",
            "role": "tool",
            "content": [{"text": "must not resurrect"}],
            "tool_call_id": "c1",
            "tool_name": "read",
        },
    )
    filler = [_row(f"filler-{i}", 10 + i) for i in range(30)]
    prune = tr.TranscriptEntry(
        "prune", 99.0, tr.ENTRY_PRUNE, {"target": "target", "notice": "gone"}
    )
    b_root = b.root
    _write_session(b_root, "ffffffffffee", [owner, target, *filler, prune])
    b_index = _ready(b_root, "ffffffffffee")

    detail = _read(
        b,
        session_id="ffffffffffee",
        journal_order=True,
        order_ids=["target", "missing-id"],
        peer_journal_source=True,
        source_read="payload",
        expected_journal_generation=b_index.journal_generation,
    )
    assert detail["journal_order_state"] == "ready"
    assert detail["peer_journal_source"] is True
    assert detail["source_present_ids"] == ["target"]
    assert detail["source_missing_ids"] == ["missing-id"]
    assert detail["source_unavailable_ids"] == []
    assert detail["source_remaining_ids"] == []
    # The companion owning row is a NAMED dependency and its conversation body
    # rides dependency_entries; the effective prune is NAMED TOO (it is what
    # settles the served body) but its own body must NOT be served — it is not
    # a conversation row, and a prune body on the wire would leak removal
    # metadata. Nothing is smuggled among requested rows.
    assert set(detail["source_dependency_ids"]) == {"call-owner", "prune"}
    assert [row["id"] for row in detail["entries"]] == ["target"]
    assert {row["id"] for row in detail["dependency_entries"]} == {"call-owner"}
    assert "prune" not in {row["id"] for row in detail["dependency_entries"]}
    entry = detail["entries"][0]
    assert entry["ts_source"] == "entry" and entry["journal_seq"] == b_index.journal_rows["target"]
    # The prune applies: the SETTLED body (the notice) rides the wire, and the
    # removed text does not.
    assert "must not resurrect" not in json.dumps(detail["entries"])
    assert "gone" in json.dumps(entry["payload"])
    assert detail["journal_ordinals"]["target"] == b_index.journal_rows["target"]
    assert relay.session_history_reply_bytes(1, detail) <= relay.SESSION_HISTORY_REPLY_BUDGET_BYTES


def test_payload_requires_ids_expected_and_the_declaration(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    _ready(b.root)
    with pytest.raises(relay.MeshRefusal, match="payload requires order_ids"):
        _read(
            b,
            journal_order=True,
            peer_journal_source=True,
            source_read="payload",
            expected_journal_generation="g",
        )
    with pytest.raises(relay.MeshRefusal, match="payload requires expected_journal_generation"):
        _read(
            b, journal_order=True, order_ids=["x"], peer_journal_source=True, source_read="payload"
        )
    with pytest.raises(relay.MeshRefusal, match="requires peer_journal_source"):
        _read(
            b,
            journal_order=True,
            order_ids=["x"],
            source_read="payload",
            expected_journal_generation="g",
        )
    with pytest.raises(relay.MeshRefusal, match="source_read must be one of"):
        _read(b, journal_order=True, peer_journal_source=True, source_read="bodies")
    # The dispatcher's own lane owns availability; the handler must refuse it
    # rather than answer a page.
    with pytest.raises(relay.MeshRefusal, match="deferred source lane"):
        _read(b, journal_order=True, peer_journal_source=True, source_read="availability")


def test_expected_generation_shape_rejected(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    with pytest.raises(relay.MeshRefusal, match="expected_journal_generation must be"):
        _read(
            b,
            journal_order=True,
            peer_journal_source=True,
            source_read="metadata",
            expected_journal_generation="x" * 129,
        )
    with pytest.raises(relay.MeshRefusal, match="peer_journal_source must be"):
        _read(b, journal_order=True, peer_journal_source="yes")


def test_purpose_shape_rejected_on_the_control_boundary(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    _ready(b.root)
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    try:
        for fields, match in (
            ({"source_read": "bodies", "peer_journal_source": True}, "source_read must be"),
            ({"source_read": "payload", "peer_journal_source": True}, "payload requires"),
            ({"peer_journal_source": "yes"}, "peer_journal_source must be"),
            ({"peer_journal_source": True, "expected_journal_generation": ""}, "must be"),
        ):
            refused = _call(
                a.root,
                "peer_session_history",
                peer=b.identity.device_id,
                session_id=SESSION,
                **fields,
            )
            assert refused.get("op") == "error", refused
            assert match in str(refused.get("message") or ""), refused
    finally:
        link.close()


def test_control_boundary_needs_the_owners_link_capability(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owner relay that never advertised the string sees NO new fields."""
    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    _ready(b.root)

    class _OldLink:
        link_id = "old"
        capabilities = frozenset({"mesh-net-v1"})

    monkeypatch.setattr(a, "_ensure_link", lambda device_id: _OldLink())
    monkeypatch.setattr(a, "_resolve_peer", lambda peer: "device-b")
    forwarded: list[dict[str, Any]] = []

    def record(op: str, peer: str, **fields: Any) -> Any:
        forwarded.append({"op": op, **fields})
        # The owner's legacy page, exactly as an old relay answers it.
        return {
            "entries": [{"id": "x"}],
            "has_more": False,
            "cursor_missing": False,
            "has_newer": None,
        }

    monkeypatch.setattr(a, "_local_peer_call", record)
    # A PAGE read still SERVES ITS PAGE: the new fields are simply not sent, so
    # the owner answers today's bytes and no complete request is composed.
    page = relay.RelayServer._ctl_peer_history(
        a,
        {
            "peer": "device-b",
            "session_id": SESSION,
            "journal_order": True,
            "order_ids": ["x"],
            "peer_journal_source": True,
        },
    )
    assert page["entries"] == [{"id": "x"}]
    assert "peer_journal_source" not in forwarded[-1]
    assert "source_read" not in forwarded[-1]
    # A PURPOSE cannot be served without the owner's complete support: the
    # bounded unsupported answer, and NO hop is spent for it.
    spent = len(forwarded)
    for fields in (
        {"source_read": "metadata", "order_ids": ["x"]},
        {"source_read": "payload", "order_ids": ["x"], "expected_journal_generation": "g"},
        {"source_read": "availability"},
    ):
        answer = relay.RelayServer._ctl_peer_history(
            a,
            {"peer": "device-b", "session_id": SESSION, "peer_journal_source": True, **fields},
        )
        assert answer["journal_order_state"] == "unsupported", answer
        assert answer["peer_journal_source"] is False, answer
        if fields["source_read"] == "availability":
            assert answer["source_available"] is False
    assert len(forwarded) == spent


# ---------------------------------------------------------------------------
# The deferred availability lane, on the real pair
# ---------------------------------------------------------------------------


def test_availability_lane_answers_ready_without_a_runtime(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, ["", "", ""])
    wake_store.write_entry(
        b.root,
        SESSION,
        cwd="",
        schedules=[{"id": "no-fire", "state": "scheduled"}],
        preserve={"stopped_at": 1234},
    )
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    try:
        reply = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id=SESSION,
            peer_journal_source=True,
            source_read="availability",
        )
        assert reply.get("op") == "ack", reply
        detail = reply["detail"]
        assert detail["source_available"] is True
        assert detail["journal_order_state"] == "ready"
        assert detail["peer_journal_source"] is True
        # Never generation authority: no generation or rows may ride it.
        assert "journal_generation" not in detail
        assert "entries" not in detail
        # STOPPED OWNER PRESERVED: no runtime artefact, stop marker intact.
        assert not list(b.root.glob("sessions/*/runtime*"))
        stopped = wake_store.read_entry(b.root, SESSION)
        assert stopped is not None and stopped["stopped_at"] == 1234
        # The lane settles: no waiter is left registered.
        assert b._source_wait_total == 0
    finally:
        link.close()


def test_availability_lane_is_bounded_and_cleans_up(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 0.2)

    def held(*args: Any, **kwargs: Any) -> Any:
        # Created on the PAGE loop the flight runs on, so the wait is a real
        # bound firing rather than a cross-loop error settling it early.
        return asyncio.get_running_loop().create_future(), True

    monkeypatch.setattr(ti, "start_refresh", held)
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    try:
        reply = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id=SESSION,
            peer_journal_source=True,
            source_read="availability",
        )
        assert reply.get("op") == "ack", reply
        assert reply["detail"]["source_available"] is False
        assert reply["detail"]["journal_order_state"] == "unavailable"
        assert b._source_wait_total == 0
    finally:
        link.close()


def test_availability_coalesces_one_join_per_wave(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time as time_module

    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 0.5)
    calls: list[str] = []

    def held(root: Path, session_id: str) -> Any:
        calls.append(session_id)
        return asyncio.get_running_loop().create_future(), True

    monkeypatch.setattr(ti, "start_refresh", held)
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    results: list[dict[str, Any]] = []

    def ask() -> None:
        results.append(
            _call(
                a.root,
                "peer_session_history",
                peer=b.identity.device_id,
                session_id=SESSION,
                peer_journal_source=True,
                source_read="availability",
            )
        )

    try:
        threads = [threading.Thread(target=ask) for _ in range(2)]
        for thread in threads:
            thread.start()
        # BOTH asks land in the ONE wave: one refresh join between them.
        deadline = time_module.monotonic() + 2.0
        while b._source_wait_total < 2 and time_module.monotonic() < deadline:
            time_module.sleep(0.01)
        assert b._source_wait_total == 2
        for thread in threads:
            thread.join(timeout=6.0)
        assert calls == [SESSION], calls
        assert len(results) == 2
        for reply in results:
            assert reply["detail"]["source_available"] is False
            assert reply["detail"]["journal_order_state"] == "unavailable"
        assert b._source_wait_total == 0
    finally:
        link.close()


def test_availability_cap_answers_bounded_busy(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time as time_module

    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 0.6)
    monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_MAX_PENDING", 1)
    monkeypatch.setattr(
        ti,
        "start_refresh",
        lambda *args, **kwargs: (asyncio.get_running_loop().create_future(), True),
    )
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    first: list[dict[str, Any]] = []

    def ask() -> None:
        first.append(
            _call(
                a.root,
                "peer_session_history",
                peer=b.identity.device_id,
                session_id=SESSION,
                peer_journal_source=True,
                source_read="availability",
            )
        )

    try:
        thread = threading.Thread(target=ask)
        thread.start()
        deadline = time_module.monotonic() + 2.0
        while b._source_wait_total < 1 and time_module.monotonic() < deadline:
            time_module.sleep(0.01)
        assert b._source_wait_total == 1
        # Cap 1 SATURATED: the second ask is answered bounded-busy, never queued.
        second = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id=SESSION,
            peer_journal_source=True,
            source_read="availability",
        )
        assert second.get("op") == "ack", second
        assert second["detail"]["source_available"] is False
        assert second["detail"]["reason"] == "busy"
        thread.join(timeout=6.0)
        assert first and first[0]["detail"]["source_available"] is False
        assert b._source_wait_total == 0
    finally:
        link.close()


def test_duplicate_availability_request_registers_once(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    _a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 0.3)
    monkeypatch.setattr(
        ti,
        "start_refresh",
        lambda *args, **kwargs: (asyncio.get_running_loop().create_future(), True),
    )
    frame = {
        "session_id": SESSION,
        "req": 7,
        "peer_journal_source": True,
        "source_read": "availability",
    }
    link = _StubLink(alive=True)
    b._ensure_link = lambda device_id: link  # type: ignore[method-assign]
    b._resolve_peer = lambda peer: "device-b"  # type: ignore[method-assign]
    first = relay.RelayServer._dispatch_source_availability(b, cast(Any, link), dict(frame))
    assert first is None  # deferred: the reply leaves later
    assert b._source_wait_total == 1
    second = relay.RelayServer._dispatch_source_availability(b, cast(Any, link), dict(frame))
    assert second is None  # the duplicate is suppressed, not re-registered
    assert b._source_wait_total == 1


class _StubLink:
    def __init__(self, *, alive: bool = True, link_id: str = "stub") -> None:
        self.alive = alive
        self.link_id = link_id
        self.context = LinkContext(
            link_id=link_id,
            device_id="device-a",
            instance_id="instance-a",
            network_id="n",
            epoch=1,
            capabilities=frozenset({"read"}),
            phase="member",
        )
        self.sent: list[dict[str, Any]] = []

    def send(self, frame: dict[str, Any]) -> bool:
        self.sent.append(frame)
        return True


def test_availability_answer_rechecks_after_the_wait(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    frame = {
        "session_id": SESSION,
        "req": 1,
        "peer_journal_source": True,
        "source_read": "availability",
    }

    # A closed link gets NOTHING — there is no reader left to answer.
    dead = _StubLink(alive=False)
    relay.RelayServer._answer_source_availability(b, cast(Any, dead), dict(frame), ready=True)
    assert dead.sent == []

    class _Permissive:
        def check(self, context: Any, candidate: dict[str, Any]) -> Any:
            return None

    class _Revoking:
        def check(self, context: Any, candidate: dict[str, Any]) -> Any:
            raise Refusal("member_revoked", "this device's grant was revoked")

    # The stub links carry no real grant, so the authoriser is stubbed for this
    # test's three arms; every OTHER check still runs for real.
    original = b.authorizer
    b.authorizer = _Permissive()  # type: ignore[assignment]
    try:
        # A revoked grant gets its ONE bounded answer (a refusal), not silence.
        b.authorizer = _Revoking()  # type: ignore[assignment]
        revoked = _StubLink()
        relay.RelayServer._answer_source_availability(
            b, cast(Any, revoked), dict(frame), ready=True
        )
        assert len(revoked.sent) == 1 and revoked.sent[0]["op"] == "error"
        # A session this device no longer holds is refused the same way.
        b.authorizer = _Permissive()  # type: ignore[assignment]
        moved = _StubLink()
        relay.RelayServer._answer_source_availability(
            b, cast(Any, moved), {**frame, "session_id": "fffffffffff0"}, ready=True
        )
        assert len(moved.sent) == 1 and moved.sent[0]["op"] == "error"
        served = _StubLink()
        relay.RelayServer._answer_source_availability(b, cast(Any, served), dict(frame), ready=True)
        assert len(served.sent) == 1
        body = served.sent[0]
        assert body["op"] == "ack" and body["detail"]["source_available"] is True
        assert body["detail"]["peer_journal_source"] is True
    finally:
        b.authorizer = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Denied and wrong-session reads leak no metadata
# ---------------------------------------------------------------------------


def test_denied_and_wrong_session_purposes_leak_no_metadata(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    _ready(b.root)
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    try:
        for purpose in ("metadata", "payload"):
            wrong = _call(
                a.root,
                "peer_session_history",
                peer=b.identity.device_id,
                session_id="ffffffffffe1",
                peer_journal_source=True,
                source_read=purpose,
                order_ids=["x"],
                expected_journal_generation="g",
            )
            assert wrong.get("op") == "error", wrong
            assert "journal_generation" not in json.dumps(wrong)
        availability = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id="ffffffffffe1",
            peer_journal_source=True,
            source_read="availability",
        )
        assert availability.get("op") == "error", availability
        assert "source_available" not in json.dumps(availability)
        # The real grant check runs before owner metadata construction.
        from local_operator.network import store as network_store

        local = network_store.load(record.network_id, b.root)
        assert local is not None
        member = next(row for row in local.members if row.device_id == a.identity.device_id)
        member.capabilities = []
        network_store.save(local, b.root)
        denied = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id=SESSION,
            peer_journal_source=True,
            source_read="metadata",
            order_ids=["x"],
        )
        assert denied.get("op") == "error", denied
        assert "journal_generation" not in json.dumps(denied)
    finally:
        link.close()


# ---------------------------------------------------------------------------
# The bridge's wake machinery
# ---------------------------------------------------------------------------


def _bridge(tmp_path: Path, remote: bool = True) -> Any:
    from local_operator.resume import SessionRow
    from local_operator.server.utils.desktop_sessions import DesktopSessionBridge

    row = None
    if remote:
        row = SessionRow(
            id=SESSION,
            mtime=0,
            name="",
            live_state="idle",
            locality="remote",
            owner_device="device-b",
            owner_device_name="device-b",
            reachable=True,
            unreachable_reason="",
        )
    bridge = DesktopSessionBridge(tmp_path, SESSION, "", remote_row=row)
    return bridge, row


def test_bridge_wake_requires_the_subscriber_declaration(tmp_path: Path) -> None:
    bridge, _row = _bridge(tmp_path)
    legacy = bridge.subscribe(journal_order=True)
    complete = bridge.subscribe(journal_order=True, peer_journal_source=True)
    assert legacy.peer_journal_source is False and complete.peer_journal_source is True
    bridge.publish("event", {"type": "journal_order_ready", "peer_source": True})
    frames = []
    while not complete.queue.empty():
        frames.append(complete.queue.get_nowait()[0])
    assert any(f["payload"].get("peer_source") is True for f in frames), frames
    assert legacy.queue.empty(), "a legacy subscriber must never meet a peer wake"
    # And a LOCAL wake (no peer marker) still reaches the legacy subscriber,
    # exactly as before this change.
    bridge.publish("event", {"type": "journal_order_ready", "rows": []})
    local_frames = []
    while not legacy.queue.empty():
        local_frames.append(legacy.queue.get_nowait()[0])
    assert any(f["payload"].get("type") == "journal_order_ready" for f in local_frames)


@pytest.mark.asyncio
async def test_bridge_flight_publishes_wake_then_respects_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, row = _bridge(tmp_path)
    sub = bridge.subscribe(journal_order=True, peer_journal_source=True)
    states = iter(["ready", "unavailable"])

    def outcome(root: Path, *, device_id: str, session_id: str) -> str:
        return next(states)

    monkeypatch.setattr(projection, "peer_source_availability", outcome)
    bridge._note_source_availability("building")
    assert bridge._source_wait_task is not None
    await bridge._source_wait_task
    frames = []
    while not sub.queue.empty():
        frames.append(sub.queue.get_nowait()[0])
    assert len(frames) == 1
    payload = frames[0]["payload"]
    assert payload["type"] == "journal_order_ready" and payload["peer_source"] is True
    # METADATA-ONLY: no rows, no generation (availability is not authority).
    assert "rows" not in payload and "journal_generation" not in payload
    # A failed wave is NOT re-armed until the state moves...
    bridge._note_source_availability("building")
    assert bridge._source_wait_task is not None
    await bridge._source_wait_task
    assert bridge._source_wait_failed is True
    frames = []
    while not sub.queue.empty():
        frames.append(sub.queue.get_nowait()[0])
    assert frames and frames[0]["payload"].get("reconcile") is True
    bridge._note_source_availability("building")
    assert bridge._source_wait_failed is True
    # ...and a READY answer stands the wave down so a later rewrite may arm it.
    bridge._note_source_availability("ready")
    assert bridge._source_wait_failed is False


@pytest.mark.asyncio
async def test_bridge_flight_drops_a_moved_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, row = _bridge(tmp_path)
    sub = bridge.subscribe(journal_order=True, peer_journal_source=True)
    monkeypatch.setattr(projection, "peer_source_availability", lambda *a, **k: "ready")

    async def move_during(*args: Any, **kwargs: Any) -> str:
        # SessionRow is a NamedTuple: the handoff REPLACES the bridge's row.
        bridge.remote_row = row._replace(owner_device="device-c")
        return "ready"

    monkeypatch.setattr(asyncio, "to_thread", move_during)
    bridge._note_source_availability("building")
    await bridge._source_wait_task
    assert sub.queue.empty(), "a wake for a moved placement must not be published"


@pytest.mark.asyncio
async def test_bridge_source_read_requires_the_owner_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, _row = _bridge(tmp_path)
    page = {
        "entries": [],
        "has_more": False,
        "cursor_missing": False,
        "has_newer": None,
        "journal_order_state": "ready",
        "journal_generation": "g",
    }
    monkeypatch.setattr(projection, "peer_source_read", lambda *a, **k: page)
    mapped = await bridge._peer_source_read(
        source_read="metadata", order_ids=[], expected_generation=None
    )
    assert mapped["journal_order_state"] == "unsupported"
    assert "journal_generation" not in mapped
    # With the ACK the facts ride through and a building state arms the lane.
    monkeypatch.setattr(
        projection,
        "peer_source_read",
        lambda *a, **k: {**page, "journal_order_state": "building", "peer_journal_source": True},
    )
    mapped = await bridge._peer_source_read(
        source_read="metadata", order_ids=[], expected_generation=None
    )
    assert mapped["journal_order_state"] == "building"
    assert bridge._source_wait_task is not None


def test_history_page_dto_stays_key_free_for_old_shapes(tmp_path: Path) -> None:
    from local_operator.server.models.desktop_sessions import HistoryPage

    old = HistoryPage.model_validate(
        {
            "entries": [
                {"id": "a", "ts": 1.0, "type": "message", "payload": {}, "ts_source": "entry"}
            ],
            "has_more": False,
            "cursor_missing": False,
            "has_newer": None,
        }
    )
    wire_bytes = old.model_dump(mode="json", exclude_none=True)
    for key in (
        "peer_journal_source",
        "reconcile",
        "source_present_ids",
        "source_missing_ids",
        "source_unavailable_ids",
        "source_remaining_ids",
        "source_retry_ids",
        "source_dependency_ids",
        "dependency_entries",
    ):
        assert key not in wire_bytes, key
    complete = HistoryPage.model_validate(
        {
            "entries": [],
            "has_more": False,
            "cursor_missing": False,
            "has_newer": None,
            "peer_journal_source": True,
            "journal_order_state": "building",
        }
    )
    assert complete.model_dump(mode="json", exclude_none=True)["peer_journal_source"] is True


# ---------------------------------------------------------------------------
# The route's own params
# ---------------------------------------------------------------------------


def test_route_wires_the_complete_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_sessions
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    from local_operator.session.transcript import (
        ENTRY_MESSAGE,
        TRANSCRIPT_FILENAME,
        TranscriptEntry,
    )

    token = "unit-desktop-token"
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    app = FastAPI()
    app.include_router(desktop_sessions.router)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.state.config_manager = ConfigManager(config_dir=tmp_path)

    async def run() -> None:
        session_id = await pool.create(str(tmp_path))
        directory = tmp_path / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        entry = TranscriptEntry(
            "abc000000001", 1.0, ENTRY_MESSAGE, {"role": "user", "content": "hello"}
        )
        (directory / TRANSCRIPT_FILENAME).write_text(entry.to_json() + "\n")
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            base = f"/v1/desktop/sessions/{session_id}"
            # A local read that DECLARES the complete contract changes nothing:
            # every complete key is absent without any complete answer behind it.
            declared = await client.get(
                f"{base}/history",
                params={
                    "journal_order": 1,
                    "peer_journal_source": 1,
                    "order_ids": ["abc000000001"],
                },
            )
            assert declared.status_code == 200, declared.text
            page = declared.json()["result"]
            assert "peer_journal_source" not in page
            assert "reconcile" not in page and "source_present_ids" not in page
            # A purpose against a LOCAL session is refused by name (409), never
            # answered with a second local dialect.
            refused = await client.get(
                f"{base}/history",
                params={"peer_journal_source": 1, "source_read": "metadata"},
            )
            assert refused.status_code == 409, refused.text

    asyncio.run(run())
