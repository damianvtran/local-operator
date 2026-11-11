"""S1 of mesh file transfer: ``lop network files pull`` over a REAL two-relay link.

docs/design/mesh-file-transfer.md. Every cell drives the CLI handler the person runs
(``network.cli.main`` → this device's control socket → B's relay → the peer link →
A's ``net_file_get``), on two serve-shaped relays on loopback with a real pairing. A
is the device that OWNS the session; B is the head node that pulls.

WHAT IS PINNED, in the manager's acceptance numbering:

1. a binary PNG and a log arrive byte-identical, the receipt's sha256 equal to a
   hash taken on BOTH ends, and the store sidecar carries the right mime;
2. an attachment the session's transcript references is pulled by digest, and an
   UNREFERENCED digest in the same store is refused (no probing the shared store);
3. a member narrowed to no ``view`` is refused, and so is a session A does not own —
   with nothing staged in either case;
4. traversal and confinement: ``..``, absolute, a symlink escape, a dotfile, the
   credential vocabulary, a FIFO and a directory are each refused, naming the rule,
   with nothing staged;
5. one flipped byte fails the pull CLOSED, in both shapes a flip can take (before the
   owner digests a chunk — only the whole-file hash can catch it — and after, in
   transit), with nothing landed in the store, at ``--out`` or in staging;
6. a link cut mid-transfer keeps the verified prefix, and the re-run resumes with
   ``resumed_bytes > 0`` and still verifies;
7. a file over the cap is refused before a byte moves.

Plus: an old peer (no ``mesh-file-v1``) gets the named update hint, ``--out`` never
overwrites without ``--force``, and the store's own adopt gate refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import files, store, wire
from local_operator.session.attachments import AttachmentStore
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures by import
    SESSION,
    Devices,
    _owned_session,
    devices,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair_settled

PEER = "device-a"
#: Not chunk-aligned on purpose: the last chunk is short, which is where an
#: off-by-one in the offset arithmetic would show.
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 12 + os.urandom(3 * 1024 * 1024 + 17)
LOG = "".join(f"2026-11-11T10:{i % 60:02d}:00Z worker[{i}] step ok\n" for i in range(30000))


@pytest.fixture()
def mesh(pair: Devices, monkeypatch: pytest.MonkeyPatch) -> Devices:  # noqa: F811
    """A owns SESSION with a scratchpad; B is a READ-role member (view only)."""
    _pair_settled(pair, monkeypatch, role="read")
    _owned_session(pair[0])
    _pad(pair[0]).mkdir(parents=True, exist_ok=True)
    return pair


def _pad(server: Any) -> Path:
    return Path(server.root) / "sessions" / SESSION / "scratchpad"


def _staged(server: Any) -> list[str]:
    directory = files.staging_root(Path(server.root))
    return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []


def _in_store(server: Any, payload: bytes) -> bool:
    digest = hashlib.sha256(payload).hexdigest()[:32]
    return (Path(server.root) / "attachments" / f"{digest}.bin").exists()


def _pull(
    capsys: pytest.CaptureFixture[str],
    source: str,
    *,
    out: Path | None = None,
    force: bool = False,
    session: str = SESSION,
) -> tuple[int, dict[str, Any]]:
    """Run ``lop network files pull --json`` through the real CLI handler."""
    capsys.readouterr()
    rc = net_cli.main(
        Namespace(
            network_command="files",
            files_command="pull",
            peer=PEER,
            session=session,
            source=source,
            out=str(out) if out is not None else None,
            force=force,
            json=True,
        )
    )
    return rc, json.loads(capsys.readouterr().out)


def test_a_png_and_a_log_arrive_byte_identical_and_verified(
    mesh: Devices, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = mesh
    (_pad(server_a) / "shots").mkdir()
    (_pad(server_a) / "shots" / "frame.png").write_bytes(PNG)
    (_pad(server_a) / "run.log").write_text(LOG, encoding="utf-8")
    for source, payload, mime in (
        ("scratchpad://shots/frame.png", PNG, "image/png"),
        ("scratchpad://run.log", LOG.encode(), "text/plain"),
    ):
        out = tmp_path / Path(source).name
        rc, receipt = _pull(capsys, source, out=out)
        assert rc == 0, receipt
        landed = out.read_bytes()
        assert landed == payload, f"{source}: the bytes differ"
        owner_side = "sha256:" + hashlib.sha256(payload).hexdigest()
        head_side = "sha256:" + hashlib.sha256(landed).hexdigest()
        assert receipt["sha256"] == owner_side == head_side, receipt
        assert receipt["verified"] is True and receipt["resumed_bytes"] == 0, receipt
        assert receipt["bytes"] == len(payload) and receipt["mime"] == mime, receipt
        assert receipt["path"] == str(out), receipt
        store_file = Path(receipt["landed"]["store"])
        assert store_file.read_bytes() == payload
        sidecar = json.loads(store_file.with_suffix(".json").read_text())
        assert sidecar == {
            "digest": receipt["attachment"],
            "mime_type": mime,
            "bytes": len(payload),
        }
        assert AttachmentStore(Path(server_b.root) / "attachments").get_bytes(
            receipt["attachment"]
        ) == (payload, mime)
    assert _staged(server_b) == [], "a finished pull leaves nothing in staging"


def test_a_referenced_attachment_pulls_by_digest_and_an_unreferenced_one_is_refused(
    mesh: Devices, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = mesh
    owner_store = AttachmentStore(Path(server_a.root) / "attachments")
    referenced = owner_store.put_bytes(PNG[:200_000], "image/png")
    unreferenced = owner_store.put_bytes(b"another session's screenshot" * 100, "image/png")
    assert referenced is not None and unreferenced is not None
    transcript = Path(server_a.root) / "sessions" / SESSION / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        row = {"id": "e9", "type": "user", "content": [{"attachment": referenced.digest}]}
        handle.write(json.dumps(row) + "\n")

    rc, receipt = _pull(capsys, f"attachment:{referenced.digest}", out=tmp_path / "ref.png")
    assert rc == 0, receipt
    assert (tmp_path / "ref.png").read_bytes() == PNG[:200_000]
    assert receipt["attachment"] == referenced.digest and receipt["mime"] == "image/png"

    rc, refused = _pull(capsys, unreferenced.digest, out=tmp_path / "probe.png")
    assert rc == 1 and refused["code"] == "not_referenced", refused
    assert not (tmp_path / "probe.png").exists()
    assert not _in_store(server_b, b"another session's screenshot" * 100)
    assert _staged(server_b) == []


def test_no_view_and_a_session_not_owned_are_refused_with_nothing_staged(
    mesh: Devices, capsys: pytest.CaptureFixture[str]
) -> None:
    server_a, server_b, _host, _port = mesh
    (_pad(server_a) / "secret-plan.txt").write_text("not for a list-only member")
    # A session that exists on B's side of the question only: A never owned it.
    rc, refused = _pull(capsys, "scratchpad://secret-plan.txt", session="0123456789ab")
    assert rc == 1, refused
    # The chokepoint's codes deliberately do not cross the wire (wire.refusal_frame),
    # so the owner's SENTENCE is the evidence of which rule fired.
    assert "does not live on this device" in refused["message"], refused

    record = store.list_networks(server_a.root)[0]
    with store.mutate(record.network_id, server_a.root) as mutable:
        member = mutable.member(server_b.identity.device_id)
        assert member is not None
        member.capabilities = ["list"]
        store.save(mutable, server_a.root)
    rc, refused = _pull(capsys, "scratchpad://secret-plan.txt")
    assert rc == 1, refused
    assert "does not hold the 'view' capability" in refused["message"], refused
    assert _staged(server_b) == [] and not _in_store(server_b, b"not for a list-only member")


def test_traversal_secrets_and_special_files_are_refused_by_rule(
    mesh: Devices, capsys: pytest.CaptureFixture[str]
) -> None:
    server_a, server_b, _host, _port = mesh
    pad = _pad(server_a)
    secrets_dir = Path(server_a.root) / "secrets"
    secrets_dir.mkdir(exist_ok=True)
    (secrets_dir / "master.key").write_text("never leaves")
    (pad / "escape").symlink_to(secrets_dir / "master.key")
    (pad / ".env").write_text("API_KEY=x")
    (pad / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
    (pad / "folder").mkdir()
    os.mkfifo(pad / "pipe")
    (Path(server_a.root) / "sessions" / "neighbour.txt").write_text("outside the pad")

    cases = {
        "scratchpad://../neighbour.txt": "outside_scratchpad",
        "scratchpad:///etc/hosts": "outside_scratchpad",
        "scratchpad://escape": "outside_scratchpad",
        "scratchpad://.env": "outside_scratchpad",
        "scratchpad://id_rsa": "sensitive",
        "scratchpad://folder/": "not_a_file",
        "scratchpad://pipe": "unsafe_path",
    }
    for source, code in cases.items():
        rc, refused = _pull(capsys, source)
        assert rc == 1 and refused["code"] == code, (source, refused)
        assert _staged(server_b) == [], source
    # NOT a scratchpad URL: refused by the CLI before the relay is asked, and by the
    # owner too when a caller skips the CLI (the CLI check is a courtesy only).
    rc, refused = _pull(capsys, "/etc/hosts")
    assert rc == 1 and refused["code"] == "bad_request", refused
    with pytest.raises(files.MeshRefusal) as raised:
        files.pull_file(server_b, peer=PEER, session_id=SESSION, path="/etc/hosts")
    assert raised.value.code == "outside_scratchpad", raised.value.sentence
    assert _staged(server_b) == []


@pytest.mark.parametrize("where", ["before_the_chunk_digest", "in_transit"])
def test_one_flipped_byte_fails_closed_with_nothing_landed(
    mesh: Devices,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    where: str,
) -> None:
    server_a, server_b, _host, _port = mesh
    (_pad(server_a) / "frame.png").write_bytes(PNG)
    fetched_from_zero: list[int] = []
    if where == "before_the_chunk_digest":
        # The owner's own read is damaged, so its chunk digest AGREES with the bad
        # bytes: only the whole-file hash announced at ``stat`` can catch this.
        real_read = files._read_span

        def flipped_read(fd: int, offset: int, limit: int) -> bytes:
            span = real_read(fd, offset, limit)
            if offset == 0:
                fetched_from_zero.append(1)
            if offset == files.FILE_CHUNK_BYTES:
                return span[:10] + bytes([span[10] ^ 0x01]) + span[11:]
            return span

        monkeypatch.setattr(files, "_read_span", flipped_read)
    else:
        serve = server_a._handlers["net_file_get"]  # noqa: SLF001

        def tampered(link: Any, frame: dict[str, Any]) -> Any:
            reply = serve(link, frame)
            assert reply is not None
            if frame.get("phase") == "fetch" and int(frame.get("offset") or 0) == 0:
                fetched_from_zero.append(1)
            if frame.get("phase") == "fetch" and int(frame["offset"]) == files.FILE_CHUNK_BYTES:
                data = bytearray(base64_decode(reply["data"]))
                data[10] ^= 0x01
                reply = {**reply, "data": base64_encode(bytes(data))}
            return reply

        monkeypatch.setitem(server_a._handlers, "net_file_get", tampered)  # noqa: SLF001

    out = tmp_path / "frame.png"
    rc, refused = _pull(capsys, "scratchpad://frame.png", out=out)
    assert rc == 1 and refused["code"] == "integrity_failed", refused
    assert len(fetched_from_zero) == 2, "one retry from zero, then a refusal"
    assert not out.exists(), "--out was written from unverified bytes"
    assert not _in_store(server_b, PNG), "unverified bytes were filed in the store"
    assert list((Path(server_b.root) / "attachments").glob("*.bin")) == []
    assert _staged(server_b) == [], "the damaged part was left behind"


def test_a_link_cut_mid_transfer_resumes_from_the_verified_prefix(
    mesh: Devices,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, _host, _port = mesh
    (_pad(server_a) / "frame.png").write_bytes(PNG)
    serve = server_a._handlers["net_file_get"]  # noqa: SLF001
    phases: list[tuple[str, int]] = []

    def cut_after_three_chunks(link: Any, frame: dict[str, Any]) -> Any:
        phase = str(frame.get("phase"))
        phases.append((phase, int(frame.get("offset") or frame.get("prefix_bytes") or 0)))
        if (
            phase == "fetch"
            and int(frame["offset"]) == 3 * files.FILE_CHUNK_BYTES
            and len([p for p in phases if p[0] == "stat"]) == 1
        ):
            link.close("test: the link drops mid-transfer")
        return serve(link, frame)

    monkeypatch.setitem(server_a._handlers, "net_file_get", cut_after_three_chunks)  # noqa: SLF001
    out = tmp_path / "frame.png"
    rc, first = _pull(capsys, "scratchpad://frame.png", out=out)
    assert rc == 1 and first["code"] == "peer_unreachable", first
    assert "staged and the next pull resumes" in first["message"], first
    sha = hashlib.sha256(PNG).hexdigest()
    assert _staged(server_b) == [f"{sha}.part"], "the verified prefix was not kept"
    assert not out.exists() and not _in_store(server_b, PNG)

    rc, second = _pull(capsys, "scratchpad://frame.png", out=out)
    assert rc == 0, second
    assert second["resumed_bytes"] == 3 * files.FILE_CHUNK_BYTES, second
    assert ("verify", 3 * files.FILE_CHUNK_BYTES) in phases, "the prefix was trusted unverified"
    assert out.read_bytes() == PNG and second["verified"] is True
    assert _staged(server_b) == []


def test_over_the_cap_is_refused_before_a_byte_moves(
    mesh: Devices, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = mesh
    monkeypatch.setattr(files, "FILE_MAX_BYTES", 1024 * 1024)
    (_pad(server_a) / "big.bin").write_bytes(b"\0" * (1024 * 1024 + 1))
    rc, refused = _pull(capsys, "scratchpad://big.bin")
    assert rc == 1 and refused["code"] == "too_large", refused
    assert _staged(server_b) == []


def test_an_old_peer_gets_the_update_hint_and_is_never_asked(
    mesh: Devices, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, _host, _port = mesh
    (_pad(server_a) / "run.log").write_text(LOG)
    asked: list[dict[str, Any]] = []
    serve = server_a._handlers["net_file_get"]  # noqa: SLF001
    monkeypatch.setitem(
        server_a._handlers,  # noqa: SLF001
        "net_file_get",
        lambda link, frame: (asked.append(frame), serve(link, frame))[1],
    )
    link = server_b._ensure_link(server_a.identity.device_id)  # noqa: SLF001
    assert link is not None
    link.capabilities = frozenset(c for c in link.capabilities if c != wire.MESH_FILE_V1)
    rc, refused = _pull(capsys, "scratchpad://run.log")
    assert rc == 1 and refused["code"] == "predates_file_transfer", refused
    assert "update device-a" in refused["message"], refused
    assert asked == [], "an old peer was sent an op it would refuse as unknown"


def test_out_is_never_overwritten_without_force(
    mesh: Devices, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    server_a, _server_b, _host, _port = mesh
    (_pad(server_a) / "run.log").write_text(LOG)
    # Its own directory: ``tmp_path`` also holds the two relays' roots.
    landing = tmp_path / "out"
    landing.mkdir()
    out = landing / "run.log"
    out.write_text("the operator's own file")
    rc, refused = _pull(capsys, "scratchpad://run.log", out=out)
    assert rc == 1 and refused["code"] == "out_exists", refused
    assert out.read_text() == "the operator's own file"
    rc, receipt = _pull(capsys, "scratchpad://run.log", out=out, force=True)
    assert rc == 0, receipt
    assert out.read_text() == LOG
    assert [p.name for p in landing.iterdir()] == ["run.log"], "a temp sibling was left"


def test_the_store_adopts_only_a_file_that_hashes_to_the_announced_digest(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "attachments"
    staged = tmp_path / "staged.part"
    staged.write_bytes(b"payload")
    wrong = "sha256:" + hashlib.sha256(b"other").hexdigest()
    assert (
        AttachmentStore(store_root).adopt_file(staged, "text/plain", expected_sha256=wrong) is None
    )
    assert not store_root.exists() or list(store_root.iterdir()) == []
    right = "sha256:" + hashlib.sha256(b"payload").hexdigest()
    ref = AttachmentStore(store_root).adopt_file(staged, "text/plain", expected_sha256=right)
    assert ref is not None and ref.bytes == 7 and not staged.exists()
    assert AttachmentStore(store_root).get_bytes(ref.digest) == (b"payload", "text/plain")


def test_a_production_shaped_root_serves_its_own_pad_and_still_refuses_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's relay root IS its config dir in production, which ``is_sensitive``
    closes whole; the carve-out must open THIS session's pad and nothing more. The
    two-relay rig cannot show it (its owner root is not the ambient config dir)."""
    from local_operator.network.types import MeshRefusal
    from local_operator.session.cleanup import mark_store

    root = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    pad = root / "sessions" / SESSION / "scratchpad"
    pad.mkdir(parents=True)
    (pad.parent / "transcript.jsonl").write_text("{}\n")
    mark_store(root / "sessions")
    (pad / "frame.png").write_bytes(PNG[:64])
    (pad / "id_rsa").write_bytes(b"key")
    served = files._resolve_source(root, SESSION, {"path": "scratchpad://frame.png"})
    assert served.path == (pad / "frame.png").resolve()
    with pytest.raises(MeshRefusal) as refused:
        files._resolve_source(root, SESSION, {"path": "scratchpad://id_rsa"})
    assert refused.value.code == "sensitive"


def base64_decode(text: str) -> bytes:
    import base64

    return base64.b64decode(text)


def base64_encode(payload: bytes) -> str:
    import base64

    return base64.b64encode(payload).decode("ascii")
