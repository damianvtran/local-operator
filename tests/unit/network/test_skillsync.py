"""User skills over the mesh (``network/skillsync.py``), on two REAL relays.

What this file pins, in the operator's own acceptance terms: a clean peer gets
a skill after ONE tick of the real definitions cadence and ``skill://<name>``
resolves there through the product's own discovery; an edit on the head arrives
on the next tick; an oversize skill is reported by name, never dropped
silently; a node's own or edited skill is kept; a bundle with a traversal path
or an exec bit is refused by the receiver with nothing written; an old peer is
skipped by capability without ever being asked.

ONE PROCESS, TWO ROOTS: the fixtures are ``test_relay_e2e``'s, so every write
lands in the per-test ``root`` (an isolated HOME), never the operator's.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

from local_operator.hub_sync import skill_bundle
from local_operator.network import cli as net_cli
from local_operator.network import definitions, readiness, skillsync, store, wire
from local_operator.skills.discovery import discover_skills
from local_operator.skills.protocol import resolve_skill_url
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[Any, Any, str, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(
    root: Path, name: str, body: str = "Do the thing.", files: dict[str, bytes] | None = None
) -> Path:
    folder = root / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} for tests\n---\n\n{body}\n", encoding="utf-8"
    )
    for relative, content in (files or {}).items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return folder


def _resolves(root: Path, name: str) -> str | None:
    """``skill://<name>`` through the product's discovery + resolver for ``root``."""
    found, _warnings = discover_skills([root / "skills"], config_dir=root)
    try:
        return resolve_skill_url(f"skill://{name}", {skill.name: skill for skill in found})
    except ValueError:  # the resolver's "Unknown skill" answer
        return None


def _bind_and_start(server_b: Any, record: Any) -> str:
    host, port = server_b.bind()
    server_b.bind_control()
    server_b.start()
    live = f"{host}:{port}"
    with store.mutate(record.network_id, server_b.root) as copy:
        own = copy.member(server_b.identity.device_id)
        if own is not None:
            own.endpoints = [live]
            store.save(copy, server_b.root)
    return live


def _linked(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    live = _bind_and_start(server_b, record)
    with store.mutate(record.network_id, server_a.root) as copy:
        row = copy.member(server_b.identity.device_id)
        assert row is not None
        row.endpoints = [live]
        store.save(copy, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    return server_a, server_b


class _Clock:
    """Drives the REAL ``DefinitionsSyncer.tick`` past its per-member floors."""

    def __init__(self, server: Any) -> None:
        self.syncer = definitions.DefinitionsSyncer(server)
        self.now = 1_000_000.0

    def tick(self) -> None:
        self.now += definitions.STATE_MIN_INTERVAL_S + 1
        self.syncer.tick(now=self.now)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser


def _run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    capsys.readouterr()
    rc = net_cli.main(_parser().parse_args(["network", "skills", *argv, "--json"]))
    return rc, json.loads(capsys.readouterr().out)


def _zip(entries: list[tuple[str, bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content, mode in entries:
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def _hostile_payload(entries: list[tuple[str, bytes, int]]) -> dict[str, Any]:
    data = _zip(entries)
    return {
        "kind": skillsync.BUNDLE_KIND,
        "version": skillsync.BUNDLE_VERSION,
        "name": "evil",
        "bundle": {
            "format": "zip",
            "hash": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "uncompressed_size": sum(len(c) for _p, c, _m in entries),
            "file_count": len(entries),
            "entrypoint": "SKILL.md",
            "tree_hash": "0" * 64,
        },
        "data": base64.b64encode(data).decode("ascii"),
    }


_SKILL_MD = b"---\nname: evil\ndescription: evil\n---\nhi\n"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_both_ends_serve_the_op_and_advertise_the_capability(
    request: pytest.FixtureRequest,
) -> None:
    server_a, server_b, _host, _port = request.getfixturevalue("devices")
    for server in (server_a, server_b):
        assert "net_skills" in server._handlers  # noqa: SLF001 — the slice's seat
        assert "skills_sync" in server._local_slice_handlers  # noqa: SLF001
    assert wire.SKILLS_SYNC_V1 in wire.LINK_CAPABILITIES
    assert skillsync.mesh_tick_step in definitions._tick_steps()


# ---------------------------------------------------------------------------
# The acceptance matrix over a real link
# ---------------------------------------------------------------------------


def test_a_clean_peer_gets_a_skill_after_one_tick_and_an_edit_on_the_next(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b = _linked(request, monkeypatch)
    # A +x script ships AS DATA (the head's minerva-doctor carries one).
    folder = _write_skill(
        server_a.root, "deploy-notes", files={"scripts/run.sh": b"#!/bin/sh\necho hi\n"}
    )
    os.chmod(folder / "scripts" / "run.sh", 0o755)
    assert _resolves(server_b.root, "deploy-notes") is None

    clock = _Clock(server_a)
    clock.tick()

    landed = server_b.root / "skills" / "deploy-notes"
    assert (landed / "SKILL.md").read_text(encoding="utf-8").endswith("Do the thing.\n")
    assert (landed / "scripts" / "run.sh").read_bytes() == b"#!/bin/sh\necho hi\n"
    # Remote content is never executable merely because it arrived.
    assert os.stat(landed / "scripts" / "run.sh").st_mode & 0o111 == 0
    resolved = _resolves(server_b.root, "deploy-notes")
    assert resolved is not None and "Do the thing." in resolved
    # The peer records provenance, so it never re-sends the mirror as its own.
    assert [s.name for s in skillsync.authored_skills(server_b.root)] == []

    _write_skill(server_a.root, "deploy-notes", body="Do the NEW thing.")
    clock.tick()
    resolved = _resolves(server_b.root, "deploy-notes")
    assert resolved is not None and "Do the NEW thing." in resolved

    # Idempotent: a further tick finds the peer in sync and rewrites nothing.
    before = os.stat(landed / "SKILL.md").st_mtime_ns
    clock.tick()
    assert os.stat(landed / "SKILL.md").st_mtime_ns == before
    receipt = skillsync.read_receipts(server_a.root)[server_b.identity.device_id]
    assert receipt["code"] == "in_sync", receipt
    server_b.stop()


def test_an_oversize_skill_is_reported_by_name_not_dropped_silently(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_a, server_b = _linked(request, monkeypatch)
    monkeypatch.setattr(skillsync, "MAX_SKILL_ARCHIVE_BYTES", 2048)
    _write_skill(server_a.root, "small-one")
    _write_skill(server_a.root, "huge-one", files={"assets/blob.bin": os.urandom(8192)})

    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 0, payload
    row = payload["peers"][0]
    assert [item["name"] for item in row["installed"]] == ["small-one"], row
    assert [item["name"] for item in row["skipped"]] == ["huge-one"], row
    assert "per-skill mesh cap" in row["skipped"][0]["reason"]
    assert "huge-one" in row["message"], row
    assert not (server_b.root / "skills" / "huge-one").exists()

    rc, state = _run_json(capsys, "state")
    huge = next(item for item in state["skills"] if item["name"] == "huge-one")
    assert huge["status"] == "too_large", huge

    # `lop network ready`'s row names it too (composed from B's real answer).
    facts = {"skills": readiness.skills_fact(server_b.root)}
    member = type("M", (), {"device_id": server_b.identity.device_id})()
    ready_row = readiness.skills_row(
        member,
        facts,
        peer_label="device-b",
        local=skillsync.authored_skills(server_a.root),
        own_device_id=server_a.identity.device_id,
    )
    assert ready_row["ok"] is True, ready_row
    assert "will not travel: huge-one" in ready_row["detail"], ready_row
    assert readiness.CAPABILITY_SKILLS in readiness.NON_GATING_EQUIPMENT
    server_b.stop()


def test_a_node_owned_or_node_edited_skill_is_kept(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_a, server_b = _linked(request, monkeypatch)
    _write_skill(server_a.root, "authored-there", body="head copy")
    _write_skill(server_b.root, "authored-there", body="the node's own copy")
    _write_skill(server_a.root, "edited-there", body="v1")

    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 0, payload
    row = payload["peers"][0]
    assert [item["name"] for item in row["kept"]] == ["authored-there"], row
    assert "its own skill" in row["kept"][0]["reason"]
    assert "own copy" in (server_b.root / "skills" / "authored-there" / "SKILL.md").read_text()

    # The node edits its mirror; the head's next revision must not overwrite it.
    (server_b.root / "skills" / "edited-there" / "SKILL.md").write_text(
        "---\nname: edited-there\ndescription: edited\n---\nnode edit\n", encoding="utf-8"
    )
    _write_skill(server_a.root, "edited-there", body="v2")
    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    row = payload["peers"][0]
    kept = {item["name"]: item["reason"] for item in row["kept"]}
    assert "local edits" in kept["edited-there"], row
    assert "node edit" in (server_b.root / "skills" / "edited-there" / "SKILL.md").read_text()
    server_b.stop()


def test_an_old_peer_is_skipped_without_ever_being_asked(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Patched BEFORE pairing: a peer is "old" by what its handshake advertised.
    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(cap for cap in wire.LINK_CAPABILITIES if cap != wire.SKILLS_SYNC_V1),
    )
    server_a, server_b = _linked(request, monkeypatch)
    calls: list[int] = []
    original = server_b._handlers["net_skills"]  # noqa: SLF001
    server_b._handlers["net_skills"] = lambda link, frame: (  # noqa: SLF001
        calls.append(1),
        original(link, frame),
    )[1]
    _write_skill(server_a.root, "anything")

    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 1, payload
    assert calls == [], "the push asked a peer that never advertised skill sync"
    row = payload["peers"][0]
    assert row["code"] == "peer_too_old", row
    assert "lop-update" in row["message"]
    assert not (server_b.root / "skills" / "anything").exists()
    server_b.stop()


# ---------------------------------------------------------------------------
# The receiver refuses hostile bundles (no relay needed: the apply is the gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entries, label",
    [
        ([("SKILL.md", _SKILL_MD, 0o100644), ("../escape.txt", b"x", 0o100644)], "traversal"),
        ([("SKILL.md", _SKILL_MD, 0o100644), ("run.sh", b"#!/bin/sh\n", 0o100755)], "exec bit"),
    ],
)
def test_the_receiver_refuses_a_traversal_or_exec_bit_bundle(
    tmp_path: Path, entries: list[tuple[str, bytes, int]], label: str
) -> None:
    root = tmp_path / "peer"
    root.mkdir()
    answer = skillsync.apply_skill(root, _hostile_payload(entries), origin_device="dev-a")
    assert answer["outcome"] == "refused", (label, answer)
    assert not (root / "skills" / "evil").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not (root / "escape.txt").exists()
    # The stage is cleaned up; nothing half-extracted is left anywhere.
    stage = root / "network" / "skill-stage"
    assert not stage.exists() or not any(stage.iterdir())
    assert "evil" not in json.dumps(definitions.read_index(root).get("skills") or {})


def test_the_builder_ships_exec_bits_as_data_only_when_asked(tmp_path: Path) -> None:
    folder = _write_skill(tmp_path, "with-script", files={"scripts/x.sh": b"echo\n"})
    os.chmod(folder / "scripts" / "x.sh", 0o755)
    # The hub's default stays: an executable source is refused (§6.1).
    with pytest.raises(skill_bundle.SkillBundleError):
        skill_bundle.build_bundle(folder)
    built = skill_bundle.build_bundle(folder, executable_as_data=True)
    with zipfile.ZipFile(io.BytesIO(built.data)) as archive:
        modes = {info.filename: info.external_attr >> 16 for info in archive.infolist()}
    assert modes["scripts/x.sh"] == 0o100644
    assert built.tree_hash == skill_bundle.tree_hash_of(folder, executable_as_data=True)
    # setuid is still refused even as data.
    os.chmod(folder / "scripts" / "x.sh", 0o4755)
    with pytest.raises(skill_bundle.SkillBundleError):
        skill_bundle.build_bundle(folder, executable_as_data=True)
