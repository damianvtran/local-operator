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


# ---------------------------------------------------------------------------
# Round-1 remediation (review M1/m1/m2/m4/n1, QA Q-1..Q-5)
# ---------------------------------------------------------------------------


def _payload_from(folder: Path, name: str) -> dict[str, Any]:
    """An honest sender's frame for ``folder`` installed as ``name``."""
    built = skill_bundle.build_bundle(folder, executable_as_data=True)
    return {
        "kind": skillsync.BUNDLE_KIND,
        "version": skillsync.BUNDLE_VERSION,
        "name": name,
        "bundle": built.bundle.to_json(),
        "data": base64.b64encode(built.data).decode("ascii"),
    }


def test_a_peer_skill_cannot_shadow_the_receivers_own_by_its_frontmatter_name(
    tmp_path: Path,
) -> None:
    """M1, QA's repro: folder ``aaa`` whose SKILL.md says ``name: zzz-own``.

    Before the fix the receiver installed it as ``aaa`` and discovery then
    registered ``zzz-own`` with the PEER's description, shadowing the node's own.
    """
    receiver = tmp_path / "r"
    _write_skill(receiver, "zzz-own", body="the node's own")
    sender = tmp_path / "s"
    folder = sender / "skills" / "aaa"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: zzz-own\ndescription: the peer's impostor\n---\nhi\n", encoding="utf-8"
    )

    answer = skillsync.apply_skill(receiver, _payload_from(folder, "aaa"), origin_device="dev-a")
    assert answer["outcome"] == "refused", answer
    assert not (receiver / "skills" / "aaa").exists()
    found, _warnings = discover_skills([receiver / "skills"], config_dir=receiver)
    own = {skill.name: skill for skill in found}["zzz-own"]
    assert own.description == "zzz-own for tests"

    # The sender names it as "will not travel" instead of spending a frame.
    (row,) = [skill for skill in skillsync.local_skills(sender) if skill.name == "aaa"]
    assert row.status == "refused" and "zzz-own" in row.reason, row
    assert skillsync.state_rows(sender)[0]["status"] == "refused"


def test_a_symlinked_or_grouped_skill_is_named_never_dropped(tmp_path: Path) -> None:
    """Q-1 / Q-2: both resolve locally through discovery, so both must be named."""
    root = tmp_path / "head"
    library = tmp_path / "library" / "linked-skill"
    library.mkdir(parents=True)
    (library / "SKILL.md").write_text(
        "---\nname: linked-skill\ndescription: d\n---\nx\n", encoding="utf-8"
    )
    (root / "skills").mkdir(parents=True)
    os.symlink(library, root / "skills" / "linked-skill")
    group = root / "skills" / "my-group" / "inner-dir"
    group.mkdir(parents=True)
    (group / "SKILL.md").write_text(
        "---\nname: grouped-skill\ndescription: d\n---\nx\n", encoding="utf-8"
    )

    rows = {skill.name: skill for skill in skillsync.local_skills(root)}
    assert rows["linked-skill"].status == "refused"
    assert "not synced (symlink)" in rows["linked-skill"].reason
    assert "my-group" not in rows, "the grouping folder was named instead of its skill"
    assert rows["grouped-skill"].status == "refused"
    assert "my-group" in rows["grouped-skill"].reason
    # Named on every surface the sender reads; never followed into the state.
    assert {row["name"] for row in skillsync.state_rows(root)} >= {"linked-skill", "grouped-skill"}
    assert "linked-skill" not in skillsync.skill_state(root)["skills"]
    assert all(skill.data is None for skill in rows.values())


def test_the_old_peer_ready_row_carries_the_lop_update_remedy() -> None:
    """Q-4: the row had ``remedies: []`` while push/tick/state named ``lop-update``."""
    member = type("Member", (), {"device_id": "d_" + "b" * 32})()
    row = readiness.skills_row(member, {}, peer_label="device-b", local=[], own_device_id="d_a")
    assert any("lop-update" in remedy and "device-b" in remedy for remedy in row["remedies"]), row


def test_a_non_admin_member_neither_dials_on_create_nor_relays_the_peers_sentence(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m2 / Q-3 (and n1's real path): device-b joined as ``drive``, not ``admin``."""
    server_a, server_b = _linked(request, monkeypatch)
    calls: list[int] = []
    original = server_a._handlers["net_skills"]  # noqa: SLF001
    server_a._handlers["net_skills"] = lambda link, frame: (  # noqa: SLF001
        calls.append(1),
        original(link, frame),
    )[1]
    _write_skill(server_b.root, "from-b")
    a_id = server_a.identity.device_id

    assert skillsync.push_for_create(server_b, a_id) == {}, "a create printed a skills line"
    pushed = skillsync.push_to_peer(server_b, a_id)
    assert pushed["code"] == "no_admin", pushed
    assert "may not do that on this device" not in pushed["message"]
    assert "member grant" in pushed["message"]
    assert calls == [], "a non-admin member dialled the peer for a refusal"
    assert not (server_a.root / "skills" / "from-b").exists()
    server_b.stop()


def test_the_create_budget_covers_the_cold_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m1: a slow ``authored_skills`` (cold cache) runs INSIDE the bounded join."""
    import threading
    import time
    from types import SimpleNamespace

    release = threading.Event()

    def slow_authored(_root: Path) -> list[Any]:
        release.wait(5)
        return []

    monkeypatch.setattr(skillsync, "authored_skills", slow_authored)
    monkeypatch.setattr(skillsync, "CREATE_PUSH_BUDGET_S", 0.2)
    server = SimpleNamespace(root=tmp_path, _member_name=lambda _d: "device-b")
    started = time.monotonic()
    answer = skillsync.push_for_create(server, "d_b")  # type: ignore[arg-type]
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 2.0, elapsed
    assert answer.get("background") is True, answer


def test_push_text_agrees_with_ready_and_names_the_member(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-5: everything kept used to read "device-b holds this device's skills"."""
    server_a, server_b = _linked(request, monkeypatch)
    _write_skill(server_a.root, "both-own", body="head")
    _write_skill(server_b.root, "both-own", body="node")
    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 0, payload
    row = payload["peers"][0]
    assert row["name"], row
    assert "holds 0 of this device's 1 sendable skills" in row["message"], row
    # Q-7: the summary line agrees with the per-peer counts (it said "1 of 1
    # device holds this device's skills" beside "holds 0 of 1; 1 kept").
    capsys.readouterr()
    net_cli.main(
        _parser().parse_args(["network", "skills", "push", "--peer", server_b.identity.name])
    )
    summary = capsys.readouterr().out.splitlines()[0]
    assert summary == "0 of 1 device holds all of this device's sendable skills", summary
    net_cli.main(_parser().parse_args(["network", "skills", "state"]))
    text = capsys.readouterr().out
    assert f"last push to {row['name']}:" in text, text
    assert server_b.identity.device_id not in text, text
    server_b.stop()


def test_a_crash_leftover_stage_is_swept_but_a_live_one_is_not(tmp_path: Path) -> None:
    """m4: an apply killed before its ``finally`` left a 0600 tree forever."""
    stage = tmp_path / "network" / "skill-stage"
    stale = stage / "deadbeef" / "tree"
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text("x", encoding="utf-8")
    fresh = stage / "live"
    fresh.mkdir()
    old = 1_000_000.0
    os.utime(stage / "deadbeef", (old, old))
    sweep = skillsync._sweep_stage  # noqa: SLF001 — the crash-recovery seam under test
    removed = sweep(tmp_path, now=old + skillsync.STAGE_STALE_S + 1)
    assert removed == 1
    assert not (stage / "deadbeef").exists()
    assert fresh.exists(), "a stage younger than the cutoff belongs to a live apply"


def test_the_sweep_never_follows_a_symlinked_stage(tmp_path: Path) -> None:
    """R2-2: a linked ``network/skill-stage`` made the sweep delete in its target."""
    victim = tmp_path / "victim" / "olddir"
    victim.mkdir(parents=True)
    old = 1_000_000.0
    os.utime(victim, (old, old))
    root = tmp_path / "root"
    (root / "network").mkdir(parents=True)
    os.symlink(tmp_path / "victim", root / "network" / "skill-stage")
    sweep = skillsync._sweep_stage  # noqa: SLF001 — the crash-recovery seam under test
    assert sweep(root, now=old + skillsync.STAGE_STALE_S + 1) == 0
    assert victim.exists(), "the sweep followed a symlinked stage into its target"


def _own_skill_at(root: Path, folder: str, name: str, body: str) -> Path:
    path = root / "skills" / folder
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: receiver OWN {body}\n---\n{body}\n", encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    ("folder", "name"),
    [
        ("tools/foo", "foo"),  # grouped: deeper, so the peer's skills/foo used to win
        ("my-foo", "foo"),  # folder != name, sorting AFTER the name (peer won)
        ("zzzz-folder", "abc"),  # QA's Q-6 repro
        ("aaa-folder", "abc"),  # reverse order: own wins, but the head was told "installed"
    ],
)
def test_a_peer_skill_never_lands_beside_the_receivers_own_declared_elsewhere(
    tmp_path: Path, folder: str, name: str
) -> None:
    """R2-1 / Q-6: the keep check keyed on folders while discovery keys on names."""
    receiver = tmp_path / "r"
    _own_skill_at(receiver, folder, name, "kept")
    sender = tmp_path / "s"
    honest = _write_skill(sender, name, body="peer")

    # The receiver's state names it as its own, so the sender keeps it unasked.
    held = skillsync.skill_state(receiver)["skills"][name]
    assert held["origin"] == "" and held["folder"] == folder, held

    answer = skillsync.apply_skill(receiver, _payload_from(honest, name), origin_device="dev-a")
    assert answer["outcome"] == "conflicts", answer
    assert f"skills/{folder}" in answer["reason"], answer
    assert not (receiver / "skills" / name).exists(), "the peer's copy landed beside the own one"
    found, _warnings = discover_skills([receiver / "skills"], config_dir=receiver)
    assert {skill.name: skill for skill in found}[name].description == "receiver OWN kept"


def test_a_push_over_the_mesh_reports_an_own_skill_in_another_folder_as_kept(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-6 over two real relays: kept, never installed, and the node's own resolves."""
    server_a, server_b = _linked(request, monkeypatch)
    _write_skill(server_a.root, "abc", body="head")
    _own_skill_at(server_b.root, "zzzz-folder", "abc", "kept")
    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 0, payload
    row = payload["peers"][0]
    assert row["installed"] == [] and [kept["name"] for kept in row["kept"]] == ["abc"], row
    assert "in skills/zzzz-folder" in row["kept"][0]["reason"], row
    assert not (server_b.root / "skills" / "abc").exists()
    server_b.stop()


def test_the_own_declared_elsewhere_check_is_redecided_under_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1: an own declaration created during extraction still wins the swap."""
    receiver = tmp_path / "r"
    (receiver / "skills").mkdir(parents=True)
    sender = tmp_path / "s"
    honest = _write_skill(sender, "late", body="peer")
    real_extract = skill_bundle.extract_archive

    def _extract_then_edit(*args: Any, **kwargs: Any) -> Any:
        landed = real_extract(*args, **kwargs)
        _own_skill_at(receiver, "group/late", "late", "kept")
        return landed

    monkeypatch.setattr(skill_bundle, "extract_archive", _extract_then_edit)
    answer = skillsync.apply_skill(receiver, _payload_from(honest, "late"), origin_device="dev-a")
    assert answer["outcome"] == "conflicts", answer
    assert not (receiver / "skills" / "late").exists()


def test_a_grouped_twin_of_a_top_level_skill_is_named_as_shadowed(tmp_path: Path) -> None:
    """R2-3: "move it to skills/foo" pointed at the folder the twin already holds."""
    root = tmp_path / "head"
    _write_skill(root, "foo")
    _own_skill_at(root, "tools/foo", "foo", "grouped")
    rows = [skill for skill in skillsync.local_skills(root) if skill.name == "foo"]
    assert {row.status for row in rows} == {"ready", "refused"}, rows
    (grouped,) = [row for row in rows if row.status == "refused"]
    assert "shadowed here by ~/.local-operator/skills/foo" in grouped.reason, grouped
    assert "move it" not in grouped.reason, grouped


def test_net_skills_needs_admin_at_the_authoriser() -> None:
    """n1: a ``drive`` member is refused ``net_skills``; an ``admin`` one passes."""
    from local_operator.network import types as net_types
    from tests.unit.network.test_authorizer import link, make

    authorizer, _audit = make()
    drive = link(capabilities={"list", "view", "prompt", "steer", "stop", "slash"})
    with pytest.raises(net_types.Refusal) as excinfo:
        authorizer.check(drive, {"op": "net_skills", "req": 1, "phase": "state"})
    assert excinfo.value.code == "not_authorised"
    admin = link(capabilities={"list", "view", "admin"})
    assert authorizer.check(admin, {"op": "net_skills", "req": 2, "phase": "state"}).action == (
        "net_skills"
    )
