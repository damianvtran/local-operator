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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.network import audit as audit_mod  # noqa: E402
from local_operator.network import identity, relay, store, types, wire  # noqa: E402
from scripts.visual_capture import svg_text_runs_by_row  # noqa: E402

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


def _seed_session(
    root: Path, session_id: str, title: str, *, network_id: str, home_device: str
) -> None:
    """A session B really holds, IN THE SHAPE ``/new remote <peer> <name>`` leaves.

    THE SHAPE IS THE EVIDENCE (QA round 11 Q-R11-1 = design round 2 D14). This
    used to hand-write the title sidecar under the key ``title`` while the
    product's own reader (``resume._read_title_sidecar``) reads ``text`` — so
    both seeded sessions had NO readable title, every row fell back to a bare
    id, and the committed ``static/tui-mesh-sidebar.png`` shipped a frame of the
    product painting ids where a user sees names. A capture script may not
    invent its own spelling of a format the product owns, so the title is
    written through ``resume.write_session_title`` — the same writer the create
    path uses — and the placement stamp through ``write_stamp``, mirroring
    ``RelayServer._op_session_create``. Both are the product's own formats by
    construction, so neither can drift from the reader again.

    The activity file (``transcript.jsonl``) is what makes it a row of B's own
    CATALOGUE (retention's clock), which is the half that names it from the
    sidecar. A directory with none would be a session neither device lists,
    which is the separate defect ``local_session_rows`` covers for peer-created
    sessions that have not had a turn yet — so the promptless-directory shape is
    not what this frame is about.
    """
    from local_operator.resume import write_session_title
    from local_operator.session.creation import ensure_session_created_at
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    ensure_session_created_at(directory, time.time())
    write_session_title(directory, title, user_set=True, past_names=[])
    write_stamp(
        root,
        MeshStamp(
            session_id=session_id,
            network_id=network_id,
            home_device=home_device,
            placement=SessionPlacement(
                # ``peer`` rather than ``local``, exactly as the create writes it:
                # B runs this session, but it is here because another device
                # asked for it (§5.1). A ``local`` stamp here would put the
                # frame's sessions outside the mesh membership test and the
                # picture would describe a shape the product does not produce.
                mode="peer",
                network_id=network_id,
                home_device=home_device,
                policy="pinned",
                stamp_revision=1,
            ),
            origin={"kind": "user", "source_device": "", "source_session_id": ""},
        ),
    )


def _require_rows(root: Path, titles: tuple[str, ...]) -> None:
    """Refuse to write a frame unless the producer returns these NAMES.

    A GUARD THAT COUNTS ROWS CANNOT SEE THIS CLASS OF LIE, which is exactly why
    the id-only frame survived a script written to catch it (design round 2
    D14): the producer returned the right NUMBER of rows and the wrong STRING in
    them. So the assertion is on the names, and it fails on an id (the
    nameless-row fallback that shipped) as loudly as on a missing row.
    """
    from local_operator.session.peer_rows import clear_cache, peer_session_rows

    clear_cache()
    rows = peer_session_rows(root)
    if len(rows) != len(titles):
        raise SystemExit(
            f"the producer returned {len(rows)} remote rows for a mesh holding "
            f"{len(titles)}: the frame would show a surface the product does not "
            "render, so none was written"
        )
    names = sorted(row.name for row in rows)
    if names != sorted(titles):
        raise SystemExit(
            "the producer returned rows whose names are not the ones seeded: "
            f"{names} != {sorted(titles)}. A row naming a session by its id is the "
            "shape this guard exists to catch: the frame would show bare ids where "
            "a user sees titles."
        )
    print(f"producer rows: {[(row.id, row.name, row.owner_device_name) for row in rows]}")


def _require_settled_chip(path: Path) -> None:
    """Refuse a frame whose model chip was captured MID-CONNECT (design round 4, D30).

    Re-running this rig three times at the same head produced one frame that
    differed from the committed PNG by exactly one text row —
    ``◆ connecting…  › ⌂ mesh-network`` where the settled frame reads
    ``◆ test/model``. One run in three is not a rare-enough rate for a shipped
    artifact, and nothing in the rig objected: the committed PNG happens to be a
    settled one and a re-capture could as easily have shipped the transient with
    nothing to say it had.

    Read off the exported SVG — the bytes a reviewer reads — the way the splash
    mark's census reads its own frame (`new_remote_shot._MARK_GLYPHS`), and from
    the widgets that OWN the two strings: ``status_line.ICON_MODEL`` and
    ``welcome.MODEL_PENDING`` are the band's own vocabulary, so this guard cannot
    go stale by spelling them a second time.
    """
    from local_operator.tui.widgets.status_line import ICON_MODEL
    from local_operator.tui.widgets.welcome import MODEL_PENDING

    svg = path.read_text(encoding="utf-8")
    # GROUPED BY BASELINE, because the chip is one ROW of several runs: the glyph,
    # the model label and the cwd are painted as separate spans, so a check that
    # looked inside the run holding the glyph would find no words at all and could
    # never fire on the transient it exists for (measured on this rig's own frame:
    # the only run containing ``◆`` is ``◆`` itself). The grouping itself lives in
    # `visual_capture.svg_text_runs_by_row` — this module and
    # `new_remote_shot.py` both census an export, and a private copy of the parse
    # in each is how one of them would quietly stop matching the export.
    chips = [
        "".join(runs)
        for runs in svg_text_runs_by_row(svg)
        if any(ICON_MODEL in run for run in runs)
    ]
    if not chips:
        raise SystemExit(
            f"no {ICON_MODEL!r} model chip is in {path.name}, so the band this frame "
            "is supposed to show was not painted at all"
        )
    if any(MODEL_PENDING in chip for chip in chips):
        raise SystemExit(
            f"the model chip in {path.name} reads {chips!r} — the band is still "
            "awaiting the session factory, so this frame is a mid-connect transient "
            "and not the state the artifact documents. Re-capture."
        )


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
            _seed_session(
                root_b,
                session_id,
                title,
                network_id=record_a.network_id,
                home_device=server_b.identity.device_id,
            )
        _require_rows(root_a, tuple(title for _session_id, title in SESSIONS))

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
        _require_settled_chip(out)
        print(f"wrote {out}")
        return 0
    finally:
        server_a.stop()
        server_b.stop()
        shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
