"""``workset_plan`` / ``workset_fetch`` authorisation, audit-before-bytes and resume.

Binding rules (mesh-lane owner and security lead, 2026-11-18; design §8.2, §8.3):

* every fetch is authorised against the CURRENT move's journaled artifact set — the
  in-flight ``prepared`` source entry, the AUTHENTICATED link peer equal to its
  ``to_device``, the frame's lease epoch equal to the entry's, the journal's binding
  equal to the entry's — never a directory listing (the staged directory also holds
  ``<hex>.json`` sidecars and the journal);
* a durable ``session.workset.serve`` row is on disk BEFORE the first byte of each
  artifact AND before any resumed (non-contiguous) serve; an audit that cannot be
  proven written refuses with ZERO bytes;
* the destination resumes a ``.part`` only after the owner confirms its prefix digest;
  a corrupted prefix is discarded and the pull restarts at 0.

The owner half runs for real (real git snapshot + bundle, real ``AuditLog`` file);
only the relay link is a stub, because the cells pin these functions' own checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import sync as sync_mod
from local_operator.network import workset, workset_move
from local_operator.network.audit import AuditLog
from local_operator.session.placement import write_handoff_entry
from tests.unit.network.workset_rig import GIT_ENV, Rig, make_rig

SID = "5e55a0c0ffee"
EPOCH = "epoch-1"
DEST = "d_destination"


class _Identity:
    device_id = "d_owner"


class _Server:
    def __init__(self, root: Path, home: Path) -> None:
        self.root = root
        self.workset_home = home
        self.identity = _Identity()
        self.audit = AuditLog(root)

    def _member_name(self, device_id: str) -> str:
        return "dest"


class _Link:
    def __init__(self, device_id: str) -> None:
        self.device_id = device_id


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    workset_move._SERVE_CURSOR.clear()
    return built


@pytest.fixture()
def staged(rig: Rig, tmp_path: Path) -> tuple[_Server, str]:
    """An owner with a prepared move to ``DEST`` and one planned artifact."""
    root = tmp_path / "owner"
    root.mkdir()
    server = _Server(root, rig.home)
    facts = workset.worktree_facts(rig.feat, home=rig.home, primary=rig.primary)
    manifest = workset.Manifest(
        to_device=DEST, to_name="dest", worktrees=(facts,), carries_files=True
    )
    bound, _offer = workset_move.source_prepare(
        server, SID, manifest, lease_epoch=EPOCH, to_device=DEST  # type: ignore[arg-type]
    )
    write_handoff_entry(
        root,
        SID,
        {
            "role": "source",
            "phase": "prepared",
            "mode": "move",
            "to_device": DEST,
            "lease_epoch": EPOCH,
            "instance_id": "i",
            "at": time.time(),
            "working_set": bound,
        },
    )
    frame = {"session_id": SID, "lease_epoch": EPOCH, "have": {}}
    planned = workset_move.serve_plan(server, _Link(DEST), frame)  # type: ignore[arg-type]
    assert planned["result"] == "planned", planned
    (artifact,) = planned["artifacts"]
    return server, str(artifact["digest"])


def _fetch(server: _Server, device: str = DEST, **frame: Any) -> dict[str, Any]:
    return workset_move.serve_fetch(
        server,  # type: ignore[arg-type]
        _Link(device),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": EPOCH, **frame},
    )


def _serve_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "network" / "audit.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("event") == "session.workset.serve"]


@pytest.fixture()
def spans(staged: tuple[_Server, str], monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every span the owner reads, with the number of serve rows on disk AT THAT MOMENT."""
    seen: list[tuple[int, int]] = []
    real = sync_mod.read_open_span
    root = staged[0].root

    def spy(handle: Any, *, offset: int, limit: int, size: int) -> Any:
        seen.append((offset, len(_serve_rows(root))))
        return real(handle, offset=offset, limit=limit, size=size)

    monkeypatch.setattr(sync_mod, "read_open_span", spy)
    return seen


def test_a_wrong_lease_epoch_is_refused_with_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    answer = workset_move.serve_fetch(
        server,  # type: ignore[arg-type]
        _Link(DEST),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": "epoch-0", "digest": digest, "offset": 0},
    )
    assert answer["result"] == "refused" and "data" not in answer, answer
    assert spans == [] and _serve_rows(server.root) == []


def test_a_device_that_is_not_the_destination_is_refused(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    answer = _fetch(server, device="d_third_member", digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "not_authorised", answer
    assert spans == [] and _serve_rows(server.root) == []
    plan = workset_move.serve_plan(
        server,  # type: ignore[arg-type]
        _Link("d_third_member"),  # type: ignore[arg-type]
        {"session_id": SID, "lease_epoch": EPOCH, "have": {}},
    )
    assert plan["result"] == "refused" and plan["code"] == "not_authorised", plan


@pytest.mark.parametrize(
    "name",
    [
        "sha256:" + "0" * 64,  # well-formed, not staged for this move
        "{hex}.json",  # the sidecar beside the artifact
        "journal.json",
        "sha256:../journal.json",
        "sha256:{hex}/../journal.json",
    ],
)
def test_only_a_journaled_64_hex_artifact_is_served(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], name: str
) -> None:
    server, digest = staged
    hexdigest = digest.removeprefix("sha256:")
    answer = _fetch(server, digest=name.format(hex=hexdigest), offset=0)
    assert answer["result"] == "refused" and answer["code"] == "unknown_item", answer
    assert spans == [] and _serve_rows(server.root) == []


def test_the_serve_audit_is_on_disk_before_the_first_and_every_resumed_chunk(
    staged: tuple[_Server, str], spans: list[tuple[int, int]]
) -> None:
    server, digest = staged
    first = _fetch(server, digest=digest, offset=0, limit=64)
    assert first["result"] == "chunk" and first["offset"] == 0, first
    contiguous = _fetch(server, digest=digest, offset=64, limit=64)
    assert contiguous["result"] == "chunk"
    # A relay restart (or a new pull) loses the contiguity: the next nonzero-offset
    # serve is a RESUME and must be audited again before its first byte.
    workset_move._SERVE_CURSOR.clear()
    resumed = _fetch(server, digest=digest, offset=128, limit=64)
    assert resumed["result"] == "chunk"
    # (offset, serve rows already on disk when that span was read)
    assert spans == [(0, 1), (64, 1), (128, 2)], spans
    rows = _serve_rows(server.root)
    assert [r["detail"]["offset"] for r in rows] == [0, 128]
    for row in rows:
        assert row["detail"]["to_device"] == DEST
        assert row["detail"]["lease_epoch"] == EPOCH
        assert row["detail"]["artifact"] == digest
        assert row["detail"]["path"] == "~/proj/.worktrees/feat"
        assert "data" not in row["detail"]


def test_an_audit_that_cannot_be_proven_written_serves_zero_bytes(
    staged: tuple[_Server, str], spans: list[tuple[int, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, digest = staged
    # The fsync fails (``sync`` reports it as ``degraded`` and does not count it).
    monkeypatch.setattr(server.audit, "sync", lambda: None)
    answer = _fetch(server, digest=digest, offset=0)
    assert answer["result"] == "refused" and answer["code"] == "audit_unavailable", answer
    assert "data" not in answer and spans == []


class _Transport:
    """The destination's ``ask``, wired straight to the owner's handler."""

    def __init__(self, server: _Server) -> None:
        self.server = server
        self.frames: list[dict[str, Any]] = []

    def ask(self, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        from local_operator.network.mobility import Moved

        self.frames.append(dict(frame))
        answer = workset_move.serve_fetch(self.server, _Link(DEST), frame)  # type: ignore[arg-type]
        if answer.get("result") == "refused":
            raise Moved(str(answer["code"]), str(answer["message"]))
        return answer


def _artifact_bytes(server: _Server, digest: str) -> bytes:
    hexdigest = digest.removeprefix("sha256:")
    return (workset_move.outgoing_dir(server.root, SID) / hexdigest).read_bytes()


def test_a_part_with_a_corrupted_prefix_is_discarded_and_refetched(
    staged: tuple[_Server, str], tmp_path: Path
) -> None:
    server, digest = staged
    good = _artifact_bytes(server, digest)
    into = tmp_path / "into"
    into.mkdir()
    part = into / (digest.removeprefix("sha256:") + ".part")
    part.write_bytes(b"\x00" * 50)  # same length as a real prefix, wrong bytes
    transport = _Transport(server)

    landed = workset_move._fetch_artifact(
        transport,
        session_id=SID,
        lease_epoch=EPOCH,
        digest=digest,
        expected_bytes=len(good),
        into=into,
    )

    assert landed.read_bytes() == good
    assert [f.get("prefix") for f in transport.frames if "prefix" in f] == [50]
    assert [f["offset"] for f in transport.frames if "offset" in f][0] == 0
    assert not part.exists()


def test_a_part_with_a_good_prefix_resumes_where_it_stopped(
    staged: tuple[_Server, str], tmp_path: Path
) -> None:
    server, digest = staged
    good = _artifact_bytes(server, digest)
    into = tmp_path / "into"
    into.mkdir()
    (into / (digest.removeprefix("sha256:") + ".part")).write_bytes(good[:50])
    transport = _Transport(server)

    landed = workset_move._fetch_artifact(
        transport,
        session_id=SID,
        lease_epoch=EPOCH,
        digest=digest,
        expected_bytes=len(good),
        into=into,
    )

    assert landed.read_bytes() == good
    assert [f["offset"] for f in transport.frames if "offset" in f][0] == 50
    # The resumed serve was audited before its first byte (offset 50).
    assert [r["detail"]["offset"] for r in _serve_rows(server.root)] == [50]
