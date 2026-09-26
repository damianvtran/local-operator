"""Probe: the mesh audit log's REAL I/O numbers — ``mesh-incident-response.md`` §4.8.

THE AUTHORITY FOR THE NUMBERS, not a second copy of them. The design states a cost
model (~450 B per record, ``≤ 1 write(2)/s``) and asserts four bounds; this script is
what makes both checkable on a machine, in the shape this repository already uses for
exactly this (``scripts/spend_ledger_probe.py`` is the model: name a script as the
authority rather than quoting a constant in prose).

WHAT IT DRIVES. Two relays on loopback in an ISOLATED root (``--root``, or a fresh
directory under ``$TMPDIR``), one network, two identities, a REAL handshake over a
real socket, then ``--frames`` frames down that link with a deliberate few SEMANTIC
events mixed in — the traffic a frame path must not turn into records, and the events
the log exists for. It prints frames, records, bytes, write calls and fsyncs, and
ASSERTS the bounds itself, so a regression fails here rather than needing a person to
read a number.

WHAT IT DELIBERATELY DOES NOT DRIVE: a session turn and a credential grant. Both need
a provider login, which a probe on a laptop or in CI does not have, and fabricating
one would put an invented number behind a bound — the "dead instrument returns a
reading" failure this repository has a section about. §4.8 says which half that is.

Read-only outside its own root: nothing here touches the operator's
``~/.local-operator`` (no default path derives from ``HOME``), and the temporary root
it does create is removed on the way out.

Usage::

    .venv/bin/python scripts/mesh_audit_probe.py --frames 10000 --duration 60 --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.network import audit as audit_mod  # noqa: E402
from local_operator.network import identity as identity_mod  # noqa: E402
from local_operator.network import relay, store, types, wire  # noqa: E402

#: §4.8's bounds, in one place so the probe and the design cannot drift apart.
MIN_FRAMES_PER_RECORD = 100.0
MAX_BYTES_PER_RECORD = 700.0
MAX_WRITE_CALLS_PER_MINUTE = 90.0

#: Frames queued before the probe waits for the peer to have read them. A pipelining
#: DEPTH, not a bound on the count: ``send`` blocks (and reports False) rather than
#: dropping, so a deeper batch would only make the wait coarser.
_PIPELINE_DEPTH = 256

#: How long one batch may take to reach the peer before the probe stops driving.
BATCH_DEADLINE_S = 30.0

#: The raw secret length the epoch key derives from; the probe's network has the same
#: shape as a real one.
_SECRET_BYTES = 32


def _counts(path: Path) -> tuple[int, int]:
    """``(records, bytes)`` in the live audit file.

    Counted by READING the file rather than by trusting a counter: the interesting
    failure is precisely a record that reached the disk without anything expecting it,
    so the disk is the instrument.
    """
    if not path.exists():
        return 0, 0
    records = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                records += 1
    return records, path.stat().st_size


def _records(path: Path) -> list[dict[str, Any]]:
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


def _build_pair(
    root: Path,
) -> tuple[relay.RelayServer, relay.RelayServer, types.NetworkRecord, int]:
    """Two serve-shaped relays, one network, B linked to A over a real handshake.

    A REAL MEMBER HANDSHAKE, not the human pairing ceremony: the ceremony needs two
    people at two keyboards and it is the same member row either way, while what this
    probe measures is the log's I/O. Everything below the ceremony — the epoch key
    derived from a real 32-byte secret, the MACs, the epoch gate — is the product's.
    """
    from secrets import token_bytes

    root_a = root / "a"
    root_b = root / "b"
    settings_a = relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    settings_b = relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    server_a = relay.RelayServer(
        root=root_a,
        settings=settings_a,
        identity=identity_mod.mint(root_a, name="probe-a"),
        audit=audit_mod.AuditLog(root_a),
    )
    server_b = relay.RelayServer(
        root=root_b,
        settings=settings_b,
        identity=identity_mod.mint(root_b, name="probe-b"),
        audit=audit_mod.AuditLog(root_b),
    )
    host_a, port_a = server_a.bind()
    server_a.bind_control()
    server_a.start()
    _host_b, port_b = server_b.bind()
    server_b.bind_control()
    server_b.start()

    network_id = store.new_network_id()
    secret = token_bytes(_SECRET_BYTES)
    record = types.NetworkRecord(
        network_id=network_id,
        name="probe-net",
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
        # A ``drive`` member: it cannot hold ``admin``, which is what makes the
        # definitions cadence's tier skip (the idle-write fix) visible in the same
        # run rather than only in a unit cell.
        role="drive",
        capabilities=sorted(types.capabilities_for_role("drive")),
        added_by=server_a.identity.device_id,
        added_via="invite",
        endpoints=[f"127.0.0.1:{port_b}"],
        root=root_a,
        persist=False,
    )
    store.save(record, root_a)
    store.save_secrets(
        types.SecretState(network_id=network_id, epoch=1, secret=wire.b64u(secret)), root_a
    )

    # B's own view of the same network: the joining device's record, which is the
    # member table it was admitted with plus its OWN self row.
    joined = types.NetworkRecord(
        network_id=network_id,
        name="probe-net",
        epoch=1,
        created_by=server_a.identity.device_id,
        self_device_id=server_b.identity.device_id,
        self_role="drive",
        self_capabilities=sorted(types.capabilities_for_role("drive")),
        listen={"address": "127.0.0.1", "port": port_b, "advertised": []},
    )
    joined.members = list(record.members)
    store.save(joined, root_b)
    store.save_secrets(
        types.SecretState(network_id=network_id, epoch=1, secret=wire.b64u(secret)), root_b
    )

    link, reason = server_b.dial(network_id, host=f"{host_a}:{port_a}", epoch=1)
    if link is None:
        raise SystemExit(f"the probe could not link its two relays: {reason}")
    return server_a, server_b, store.load(network_id, root_a), port_b


def _drive(
    server_b: relay.RelayServer,
    server_a: relay.RelayServer,
    network_id: str,
    *,
    frames: int,
    semantic_events: int,
) -> tuple[int, int]:
    """Push ``frames`` keepalives down the live link, then the semantic events.

    KEEPALIVES ARE THE PUREST TEST OF §4.6's FIRST ROW: they are answered on the
    reader without dispatch, so a frame path that called the audit writer shows up
    here as one record per frame — which the discrimination bound catches by orders of
    magnitude.
    """
    link = next(
        (held for held in server_b.links.values() if held.network_id == network_id and held.alive),
        None,
    )
    if link is None:
        raise SystemExit("the probe lost its link before it drove any frames")
    inbound = next(
        (
            held
            for held in server_a.links.values()
            if held.device_id == server_b.identity.device_id and held.alive
        ),
        None,
    )
    if inbound is None:
        raise SystemExit("the peer's own link is gone, so nothing would be counted")
    sent = 0
    while sent < frames:
        batch = min(_PIPELINE_DEPTH, frames - sent)
        for offset in range(batch):
            if not link.send({"op": "ping", "req": 1_000_000 + sent + offset}):
                raise SystemExit("the link refused a keepalive; its queue is wedged")
        sent += batch
        # Wait for the PEER to have read them: "the frames reached a receiver" is the
        # fact the ratio is about, and counting our own queue would report frames
        # nothing had processed.
        deadline = time.monotonic() + BATCH_DEADLINE_S
        while inbound.frames_in < sent and time.monotonic() < deadline:
            time.sleep(0.005)
        if inbound.frames_in < sent:
            break
    # NOW the semantic events, deliberately: an op this member may not dispatch, which
    # the authoriser refuses and the log must keep — and, because the cadence's tier
    # skip is in place, a probe runs at most one of them per peer rather than one per
    # 15 s.
    for index in range(semantic_events):
        link.request(
            {
                "op": "net_definitions",
                "req": 2_000_000 + index,
                "locality": "remote",
                "phase": "state",
            },
            timeout=5.0,
        )
    return int(sent), int(inbound.frames_in)


def _measure(
    server_a: relay.RelayServer,
    server_b: relay.RelayServer,
    network_id: str,
    *,
    frames: int,
    semantic_events: int,
    duration: float,
) -> dict[str, Any]:
    """Drive the pair, then report the log's own numbers for the run."""
    path = store.audit_path(server_a.root)
    log = server_a.audit
    before_records, before_bytes = _counts(path)
    before_writes = log.write_calls
    before_syncs = log.syncs
    started = time.monotonic()

    sent, frames_read = _drive(
        server_b,
        server_a,
        network_id,
        frames=frames,
        semantic_events=semantic_events,
    )
    # THE NUMERATOR IS WHAT THE PEER ACTUALLY READ, and never more than what was sent:
    # a ratio computed from queued-but-unread frames would flatter the log, and this
    # number is the one the design's "≈833 frames per record" claim rests on.
    delivered = min(sent, frames_read)
    # An IDLE TAIL, because the write-call rate is per MINUTE and a run that ends the
    # instant its last frame lands would divide by a window of milliseconds.
    elapsed_frames = max(time.monotonic() - started, 1e-3)
    if duration > elapsed_frames:
        time.sleep(duration - elapsed_frames)
    # The 1 Hz opportunistic flush is driven by the next record; ask for it explicitly
    # so a quiet tail is not reported as an unflushed buffer.
    log.flush()
    elapsed = max(time.monotonic() - started, 1e-3)

    after_records, after_bytes = _counts(path)
    records = after_records - before_records
    written = after_bytes - before_bytes
    writes = log.write_calls - before_writes
    syncs = log.syncs - before_syncs
    events = [row.get("event") for row in _records(path)[before_records:]]
    by_type: dict[str, int] = {}
    for name in events:
        by_type[str(name)] = by_type.get(str(name), 0) + 1
    rotations = by_type.get("audit_rotated", 0)
    durable_events = sum(
        count for name, count in by_type.items() if name in audit_mod.DURABLE_EVENTS
    )

    frames_per_record = (delivered / records) if records else float("inf")
    bytes_per_record = (written / records) if records else 0.0
    writes_per_minute = writes * (60.0 / elapsed)
    # THE WRITE-CALL BOUND IN TWO FORMS, because the design states one rate over a
    # minute and a run may be shorter than that: the structural form is the tick's own
    # ceiling (at most one flush per second) plus the writes a DURABLE event forces
    # through, and the rate form is that same property expressed per minute. A short
    # run is judged by the structural form and the report says which was applied —
    # asserting the per-minute number over five seconds would fail on a correct writer,
    # and asserting nothing would be the dead instrument this probe exists to avoid.
    tick_ceiling = int(elapsed) + 1
    bounds: dict[str, Any] = {
        "frames_per_record_ge_100": frames_per_record >= MIN_FRAMES_PER_RECORD,
        "bytes_per_record_le_700": bytes_per_record <= MAX_BYTES_PER_RECORD,
        "write_calls_le_tick_ceiling_plus_durable_events": writes <= durable_events + tick_ceiling,
        "fsyncs_le_durable_events_plus_rotations": syncs <= durable_events + rotations,
    }
    if elapsed >= 60.0:
        bounds["write_calls_per_minute_le_90"] = writes_per_minute <= MAX_WRITE_CALLS_PER_MINUTE
    else:
        bounds["write_calls_per_minute_bound_not_applied_below_a_minute"] = True
    return {
        "ok": all(bounds.values()),
        "bounds": bounds,
        "frames": delivered,
        "frames_sent": sent,
        "frames_read_by_peer": frames_read,
        "records": records,
        "frames_per_record": (
            round(frames_per_record, 1) if frames_per_record != float("inf") else None
        ),
        "bytes_written": written,
        "bytes_per_record": round(bytes_per_record, 1),
        "write_calls": writes,
        "write_calls_per_minute": round(writes_per_minute, 1),
        "fsyncs": syncs,
        "rotations": rotations,
        "durable_events": durable_events,
        "events_by_type": by_type,
        "elapsed_s": round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The mesh audit log's I/O numbers (§4.8).")
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--semantic-events", type=int, default=12)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = args.root or Path(tempfile.mkdtemp(prefix="mesh-audit-probe-"))
    servers: list[relay.RelayServer] = []
    try:
        server_a, server_b, record, _port_b = _build_pair(root)
        servers = [server_a, server_b]
        result = _measure(
            server_a,
            server_b,
            record.network_id,
            frames=args.frames,
            semantic_events=args.semantic_events,
            duration=args.duration,
        )
    finally:
        for server in servers:
            server.stop()
        if args.root is None:
            shutil.rmtree(root, ignore_errors=True)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
