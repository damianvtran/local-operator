"""The audit log's I/O behaviour, held to the bounds the design states (§4.8, §6.6).

WHAT THIS FILE IS FOR, and why it is not another writer unit test. The design's I/O
argument is structural: a frame path never calls the audit writer, so the log's growth
is bounded by SEMANTIC events rather than by traffic, and an idle network writes
almost nothing. Both halves have been violated in this tree — a heartbeat-style
cadence re-asking a question it could never win wrote 20 records in 300 idle seconds
(QA round 1 trust & operations, Q-R1-3) — and neither shows up in a test that drives
the writer directly. So the cells below drive the REAL path: two relays, a real link,
real frames, and the file on disk counted afterwards.

The companion instrument is ``scripts/mesh_audit_probe.py``, which reports the same
numbers for a longer run and asserts the same bounds; §4.8 names both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from local_operator.network import audit as audit_mod
from local_operator.network import definitions, identity, relay, store, types, wire

NETWORK_NAME = "audit-io"

#: How many records the 10,000-frame cell tolerates, §4.8's own number: the frames
#: driven plus the semantic events it triggers deliberately, with room for the link's
#: own lifecycle rows. At ~833 frames per record this is A7's structural claim.
FRAME_RECORD_BUDGET = 30

#: Frames the discrimination cell drives.
FRAMES = 10_000


def _records(path: Path) -> list[dict[str, Any]]:
    """Every complete record in the live file, tolerating a torn final line."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _pair(root: Path) -> tuple[relay.RelayServer, relay.RelayServer, types.NetworkRecord]:
    """Two relays on loopback, one network, B (``drive``) linked to A (``admin``).

    The member row is written through :func:`relay.admit` and the LINK is a real
    handshake over a real socket: the human pairing ceremony needs two keyboards and
    the audit's I/O does not depend on how the row was written.
    """
    from secrets import token_bytes

    root_a = root / "a"
    root_b = root / "b"
    server_a = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.mint(root_a, name="audit-a"),
        audit=audit_mod.AuditLog(root_a),
    )
    server_b = relay.RelayServer(
        root=root_b,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.mint(root_b, name="audit-b"),
        audit=audit_mod.AuditLog(root_b),
    )
    host_a, port_a = server_a.bind()
    server_a.bind_control()
    server_a.start()
    _host_b, port_b = server_b.bind()
    server_b.bind_control()
    server_b.start()
    network_id = store.new_network_id()
    secret = wire.b64u(token_bytes(32))
    record = types.NetworkRecord(
        network_id=network_id,
        name=NETWORK_NAME,
        epoch=1,
        created_by=server_a.identity.device_id,
        self_device_id=server_a.identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": port_a, "advertised": []},
    )
    relay.admit(
        record,
        device_id=server_a.identity.device_id,
        public_key=server_a.identity.public_key,
        name=server_a.identity.name,
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=server_a.identity.device_id,
        added_via="self",
        endpoints=[f"127.0.0.1:{port_a}"],
        root=root_a,
        persist=False,
    )
    relay.admit(
        record,
        device_id=server_b.identity.device_id,
        public_key=server_b.identity.public_key,
        name=server_b.identity.name,
        role="drive",
        capabilities=sorted(types.capabilities_for_role("drive")),
        added_by=server_a.identity.device_id,
        added_via="invite",
        endpoints=[f"127.0.0.1:{port_b}"],
        root=root_a,
        persist=False,
    )
    store.save(record, root_a)
    store.save_secrets(types.SecretState(network_id=network_id, epoch=1, secret=secret), root_a)
    joined = types.NetworkRecord(
        network_id=network_id,
        name=NETWORK_NAME,
        epoch=1,
        created_by=server_a.identity.device_id,
        self_device_id=server_b.identity.device_id,
        self_role="drive",
        self_capabilities=sorted(types.capabilities_for_role("drive")),
        listen={"address": "127.0.0.1", "port": port_b, "advertised": []},
    )
    joined.members = list(record.members)
    store.save(joined, root_b)
    store.save_secrets(types.SecretState(network_id=network_id, epoch=1, secret=secret), root_b)
    link, reason = server_b.dial(network_id, host=f"{host_a}:{port_a}", epoch=1)
    assert link is not None, f"the audit-io pair could not link: {reason}"
    return server_a, server_b, store.load(network_id, root_a)


def test_a_ten_thousand_frame_session_writes_fewer_than_fifty_records(root: Path) -> None:
    """§4.8's discrimination test: many frames, few semantic events.

    TEN THOUSAND KEEPALIVES ARE THE PUREST FORM OF §4.6's FIRST ROW. The peer answers
    each one on its reader without dispatch, so a call site that added an audit record
    to a frame path — the exact regression this bound exists to catch — turns the
    delta below into ~10,000 instead of a handful. The 12 requests beside them are
    deliberate SEMANTIC events (an op a ``drive`` member may not dispatch, refused and
    audited on the peer), which is the other half of the claim: the log DOES grow, by
    the events, and by nothing else.

    Measured on this head: 10,000 frames moved the peer's log by 12 records.
    """
    server_a, server_b, record = _pair(root)
    link = next(
        held
        for held in server_b.links.values()
        if held.network_id == record.network_id and held.alive
    )
    inbound = next(
        held
        for held in server_a.links.values()
        if held.device_id == server_b.identity.device_id and held.alive
    )
    path = store.audit_path(server_a.root)
    before = len(_records(path))

    # The 12 semantic events first, so the frame count below is pure frame traffic.
    for index in range(12):
        link.request(
            {
                "op": "net_definitions",
                "req": 900_000 + index,
                "locality": "remote",
                "phase": "state",
            },
            timeout=5.0,
        )
    for offset in range(FRAMES):
        assert link.send({"op": "ping", "req": 1_000_000 + offset})
    deadline = time.monotonic() + 120.0
    while inbound.frames_in < FRAMES and time.monotonic() < deadline:
        time.sleep(0.01)
    assert inbound.frames_in >= FRAMES, f"only {inbound.frames_in} frames reached the peer"

    server_a.audit.flush()
    delta = len(_records(path)) - before
    assert delta <= FRAME_RECORD_BUDGET, (
        f"{delta} records for {FRAMES} frames: something on the frame path is calling "
        f"the audit writer ({_records(path)[before:][-5:]})"
    )
    # And it is not zero either: the semantic events are the reason this log exists.
    assert delta >= 12, _records(path)[before:]
    link.close("test")
    server_a.stop()


def test_an_idle_network_writes_no_rows_for_a_member_that_cannot_hold_the_op(
    root: Path,
) -> None:
    """THE BOUNDED-IDLE PROPERTY, over a real member and an injected hour.

    Measured on this head before the fix: 300 idle seconds with one paired peer wrote
    **+21 rows / +8,805 bytes**, of which 20 were the definitions cadence retrying an
    op a ``drive`` member can never hold — every 15 s, forever, on both sides. The
    cadence is driven here through its own ``tick`` with a clock it is built to accept,
    because a cell that waited an hour would be a cell nobody runs; what it asserts is
    the property, not the wall clock: an hour of ticks writes NOTHING, because the
    answer to "may I write definitions there" does not change between them.

    TEETH: neutralise the tier check (``definitions._unholdable_capability`` returning
    ``""``, i.e. the unfixed cadence) and this cell fails — the peer's log grows by one
    refused op per tick, which is the 240-an-hour the QA measured.
    """
    server_a, server_b, record = _pair(root)
    path = store.audit_path(server_a.root)
    # FLUSH FIRST: the writer batches, and a baseline counted while the pairing's own
    # rows were still in the buffer would attribute them to the idle hour.
    server_a.audit.flush()
    before = len(_records(path))
    # A relay on B's EXISTING root, holding no link: the production shape after a
    # restart, and the shape the cadence walks member RECORDS from.
    fresh_b = relay.RelayServer(
        root=server_b.root,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=server_b.identity,
        audit=audit_mod.AuditLog(server_b.root),
    )
    syncer = definitions.DefinitionsSyncer(fresh_b)

    outcomes: list[tuple[str, str]] = []
    for tick in range(240):  # one hour at the shipped 15 s tick
        outcomes += syncer.tick(now=1_000.0 + tick * 15.0)

    assert {outcome for _device, outcome in outcomes} == {"skipped:no_admin"}, outcomes[:5]
    server_a.audit.flush()
    delta = len(_records(path)) - before
    assert delta == 0, f"an idle hour wrote {delta} rows on the peer: {_records(path)[before:][:3]}"
    server_a.stop()


def test_a_partial_final_line_is_dropped_not_fatal(root: Path) -> None:
    """§6.6's second invariant: a torn tail costs the tail, not the history.

    A crash mid-write leaves a partial line. Every complete record before it must still
    be served — an incident reconstruction that lost the whole file to one truncated
    byte is the failure this pins — and the torn line simply does not appear.
    """
    root.mkdir(parents=True, exist_ok=True)
    log = audit_mod.AuditLog(root)
    for index in range(3):
        log.record(
            audit_mod.AuditEvent(
                event="link_opened",
                detail={"role": "dialer", "epoch": index + 1, "phase": "member"},
            )
        )
    log.flush()
    path = store.audit_path(root)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "lop.mesh.audit.v1", "seq": 4, "event": "link_clo')
    rows = _records(path)
    assert [row["event"] for row in rows] == ["link_opened"] * 3, rows
    assert rows[-1]["detail"]["epoch"] == 3, rows[-1]


def test_the_writer_counts_the_calls_the_probe_reports(root: Path) -> None:
    """The probe's instrument, pinned: the counters it reads must move.

    ``write_calls_per_minute`` and ``fsyncs`` are bounds §4.8 asserts, and a bound read
    off a counter that never increments is a dead instrument — it passes forever. So
    the counter is exercised here: a durable event costs a write AND a sync, and an
    ordinary event is batched rather than written through (until the tick or the buffer
    makes it one write either way).
    """
    root.mkdir(parents=True, exist_ok=True)
    log = audit_mod.AuditLog(root)
    assert log.write_calls == 0 and log.syncs == 0
    log.record(audit_mod.AuditEvent(event="panic_raised"))
    assert log.write_calls == 1, log.write_calls
    assert log.syncs == 1, log.syncs
    for _ in range(5):
        log.record(audit_mod.AuditEvent(event="link_idle"))
    # Batched: nothing new has been written through yet beyond the durable record.
    assert log.write_calls == 1, log.write_calls
    assert log.syncs == 1, log.syncs
    log.flush()
    assert log.write_calls == 2, log.write_calls
    assert log.syncs == 1, log.syncs
    assert len(_records(store.audit_path(root))) == 6


def test_a_stream_row_publishes_at_the_state_change_and_ordinary_rows_still_batch(
    root: Path,
) -> None:
    """A7 SURVIVES THE EXCEPTION: it is per STREAM, not per row, and it is a flush.

    The audit's whole growth argument is that disk cost follows SEMANTIC events and
    never traffic, so "publish this row now" is the kind of change that can quietly
    undo it. What this cell pins is the SHAPE of the exception: an ordinary row is
    still batched (nothing written through, no sync), and a stream's lifecycle row
    costs ONE flush — one append-open and one ``.write()`` carrying whatever else was
    pending, counted by ``write_calls`` as one call however many ``write(2)`` syscalls
    the kernel splits it into — never an
    ``fsync``, because what is owed here is VISIBILITY (a reader must be able to tell
    "not flushed yet" from "no such row"), not survival of a power cut, which stays
    :data:`audit_mod.DURABLE_EVENTS`' job and is unchanged.

    Measured on the pre-fix head, the owner's close row took 14.7416 / 14.7417 /
    14.7436 s to reach the file; on this head it takes 0.0006 / 0.0015 / 0.0004 s,
    which is the whole of the difference this table makes.
    """
    root.mkdir(parents=True, exist_ok=True)
    log = audit_mod.AuditLog(root)
    path = store.audit_path(root)
    log.record(audit_mod.AuditEvent(event="link_opened", detail={"role": "dialer"}))
    # NOT WRITTEN THROUGH: the tick has not elapsed and the buffer is far from full.
    assert log.write_calls == 0, log.write_calls
    assert _records(path) == [], _records(path)

    log.record(
        audit_mod.AuditEvent(
            event="session_stream_closed",
            cause="policy",
            detail={"stream": "s_1", "peer": "d_x", "role": "owner", "cause": "viewer-left"},
        )
    )
    # ONE write, and it holds BOTH rows: the lifecycle flush publishes the batch it is
    # part of rather than a special-purpose second file, which is why the cost is one
    # write per stream and not one per row.
    assert log.write_calls == 1, log.write_calls
    assert [row["event"] for row in _records(path)] == ["link_opened", "session_stream_closed"]
    assert log.syncs == 0, "a lifecycle row was SYNCED: it owes visibility, not durability"


def test_a_reader_can_tell_an_unflushed_row_from_one_that_does_not_exist(root: Path) -> None:
    """THE DISTINCTION THE FILE CANNOT MAKE ON ITS OWN, and why it lives here.

    A reader of ``audit.jsonl`` — an operator with `jq`, or ``lop network log`` in
    another process, which cannot drain a buffer it does not own — sees the last
    PUBLISHED state. A row this writer has recorded and not yet flushed is therefore
    absent from the file, and absent is exactly what a row that was never recorded
    looks like. That ambiguity is what ``publication_of`` answers, and the cell drives
    all three of its answers on one writer: ``buffered`` for the row that exists and is
    not out yet, ``unknown`` for a number this writer never issued, ``file`` once the
    flush has happened. The first two are the pair that mattered — before this, both of
    them read as "not in the file".
    """
    root.mkdir(parents=True, exist_ok=True)
    log = audit_mod.AuditLog(root)
    path = store.audit_path(root)
    log.record(audit_mod.AuditEvent(event="link_opened", detail={"role": "dialer"}))
    held = log.recorded_through

    # EXISTS AND IS NOT OUT YET — and the file agrees it is not there, which is what
    # makes this the ambiguous case rather than a restatement of the answer.
    assert log.publication_of(held) == "buffered", log.publication_of(held)
    assert log.published_through < held <= log.recorded_through
    assert [row for row in _records(path) if row.get("seq") == held] == [], _records(path)

    # NEVER RECORDED: a DIFFERENT answer, to the same question, on the same writer.
    assert log.publication_of(held + 1) == "unknown", log.publication_of(held + 1)

    log.flush()
    assert log.publication_of(held) == "file", log.publication_of(held)
    assert log.published_through >= held
    assert log.publication_of(held + 1) == "unknown", "a flush published a row it never had"
