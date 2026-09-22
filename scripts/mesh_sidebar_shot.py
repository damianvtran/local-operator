#!/usr/bin/env python3
"""Capture the mesh sidebar over a REAL two-device mesh.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/mesh_sidebar_shot.py OUT.svg

WHY THIS SCRIPT EXISTS (QA round 10, Q-R10-1). ``sidebar_shot.py peers`` renders
the mesh tier from ``PEER_ROWS``, a hand-stamped fixture, and the README shipped
that frame as a picture of remote rows. On this branch those rows never
appeared: the producer's reply-envelope bug returned ``()`` for every device
whose peer was answering, so the published image described a surface the tree
could not render. A screenshot of a fixture cannot fail that way, which is
exactly why it did not catch it.

So this script builds the thing the frame claims to show: two config roots, two
device identities, two REAL relays on loopback, one network with both devices
admitted, a live link dialled A → B, and two sessions on B. Then it runs
``sidebar_shot.py peers-focus`` with the mesh root handed to it, so the mesh tier
in that frame comes from ``session.peer_rows.peer_session_rows`` — the
production producer, reading A's own relay over its control socket and fanning
out to B over the live link. Empty means broken: the script refuses to write a
frame at all if the producer returns no rows (see ``_require_rows``).

WHAT IT IS NOT EVIDENCE FOR, stated because a capture script's determinism
claim is load-bearing: the PAIRING here is written into both stores rather than
negotiated through the SAS ceremony (``scripts`` has no pty and no second human),
so this is not evidence about pairing. The transport is entirely real — the
dial, the handshake, the control socket, the catalogue fan-out — and that is the
path the frame is about.

Everything runs in this process except the capture, which is a CHILD process
given A's config root and left to dial A's relay the way the TUI does in
production. Both relays are stopped in ``finally``; no LaunchAgent is written.
"""

from __future__ import annotations

import secrets as _secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.network import audit as audit_mod  # noqa: E402
from local_operator.network import identity, relay, store, types, wire  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NETWORK_NAME = "devmesh"

#: The two sessions B holds, as ``(session_id, title)``. Real-length titles, so
#: the frame shows the title budget rather than a curated short name.
SESSIONS = (
    ("9f3ac1e0b7d2", "Mesh transport identity design"),
    ("4c81ba77e310", "Federated catalogue fan-out"),
)


def _network_record(
    server: relay.RelayServer, *, network_id: str, name: str, role: str
) -> types.NetworkRecord:
    return types.NetworkRecord(
        network_id=network_id,
        name=name,
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role=role,
        self_capabilities=sorted(types.capabilities_for_role(role)),
        listen={"address": "127.0.0.1", "port": server.settings.port, "advertised": []},
    )


def _admit_everyone(
    record: types.NetworkRecord, servers: list[relay.RelayServer], root: Path
) -> None:
    """Put both devices in ``record``'s table, as the pairing ceremony does."""
    for server in servers:
        role = "admin" if server is servers[0] else "drive"
        relay.admit(
            record,
            device_id=server.identity.device_id,
            public_key=server.identity.public_key,
            name=server.identity.name,
            role=role,
            capabilities=sorted(types.capabilities_for_role(role)),
            added_by=servers[0].identity.device_id,
            added_via="self",
            root=root,
            persist=False,
        )


def _seed_session(root: Path, session_id: str, title: str) -> None:
    """A session B really holds: the directory, its transcript, and its title.

    The activity file is what makes it a row of B's own catalogue (retention's
    clock); a directory with neither would be a session neither device lists,
    which is the separate defect ``local_session_rows`` now covers for
    peer-created sessions that have not had a turn yet.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    (directory / "title.json").write_text(
        f'{{"title": {title!r}, "names": [{title!r}]}}'.replace("'", '"'), encoding="utf-8"
    )


def _require_rows(root: Path, expected: int) -> None:
    from local_operator.session.peer_rows import clear_cache, peer_session_rows

    clear_cache()
    rows = peer_session_rows(root)
    if len(rows) != expected:
        raise SystemExit(
            f"the producer returned {len(rows)} remote rows for a mesh holding {expected}: "
            "the frame would show a surface the product does not render, so none was written"
        )
    print(f"producer rows: {[(row.id, row.owner_device_name) for row in rows]}")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: mesh_sidebar_shot.py OUT.svg [COLSxROWS]")
    out = Path(sys.argv[1]).resolve()
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"

    sandbox = Path(tempfile.mkdtemp(prefix="lop-mesh-shot-"))
    root_a, root_b = sandbox / "a", sandbox / "b"
    identity.mint(root_a, name="radiant-m4")
    identity.mint(root_b, name="pixel-8")
    server_a = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.load(root_a),
        audit=audit_mod.AuditLog(root_a),
    )
    server_b = relay.RelayServer(
        root=root_b,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.load(root_b),
        audit=audit_mod.AuditLog(root_b),
    )
    try:
        host_a, port_a = server_a.bind()
        server_a.bind_control()
        server_a.start()
        host_b, port_b = server_b.bind()
        server_b.bind_control()
        server_b.start()

        # ONE network, ONE epoch secret, both stores: the secret is what the
        # handshake authenticates with, so a rig that invented two would fail at
        # the first dial rather than at the frame.
        record_a = _network_record(
            server_a, network_id=store.new_network_id(), name=NETWORK_NAME, role="admin"
        )
        _admit_everyone(record_a, [server_a, server_b], root_a)
        store.save(record_a, root_a)
        state = types.SecretState(
            network_id=record_a.network_id, epoch=1, secret=wire.b64u(_secrets.token_bytes(32))
        )
        store.save_secrets(state, root_a)

        record_b = _network_record(
            server_b, network_id=record_a.network_id, name=NETWORK_NAME, role="drive"
        )
        _admit_everyone(record_b, [server_a, server_b], root_b)
        store.save(record_b, root_b)
        store.save_secrets(state, root_b)

        link, reason = server_a.dial(
            record_a.network_id, host=f"{host_b}:{port_b}", epoch=record_a.epoch
        )
        if link is None:
            raise SystemExit(f"the two devices could not link: {reason}")
        for session_id, title in SESSIONS:
            _seed_session(root_b, session_id, title)
        _require_rows(root_a, len(SESSIONS))

        env = {**__import__("os").environ, "LO_SIDEBAR_SHOT_MESH": str(root_a)}
        shot = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "sidebar_shot.py"),
                str(out),
                size,
                "peers-focus",
            ],
            cwd=REPO,
            env=env,
            check=False,
        )
        if shot.returncode != 0:
            return shot.returncode
        print(f"wrote {out}")
        return 0
    finally:
        server_a.stop()
        server_b.stop()
        shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
