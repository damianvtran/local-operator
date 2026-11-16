"""Legacy dry-run classifier, manifest and restore over SYNTHETIC roots only.

The class under test is an ABANDONMENT CANDIDATE, never "terminal": these tests
pin that the dry run is pure, lossless and exclusion-first, and that restore
recreates exact bytes/mode/mtime without ever overwriting or deleting.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator import install_transaction as tx
from local_operator import recovery_manifest as rm


def _snapshot(root: Path) -> dict[str, tuple[bytes | None, int, int]]:
    """Every path under root with bytes (files), mode and mtime_ns."""
    rows: dict[str, tuple[bytes | None, int, int]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        data = path.read_bytes() if path.is_file() and not path.is_symlink() else None
        rows[str(path.relative_to(root))] = (data, info.st_mode, info.st_mtime_ns)
    return rows


def _legacy(root: Path, identifier: str, **fields: Any) -> Path:
    directory = root / tx.OPERATION_DIR
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{identifier}.json"
    path.write_text(json.dumps({"operation_id": identifier, "phase": "partial", **fields}))
    os.chmod(path, 0o600)
    return path


def _id(index: int) -> str:
    return f"{index:032x}"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "install-root"
    (base / "generations" / "gen-current").mkdir(parents=True)
    (base / "generations" / "gen-old").mkdir()
    (base / "generations" / "gen-prepared").mkdir()
    (base / "generations" / "gen-attempt").mkdir()
    (base / "generations" / "gen-attempt" / tx.ATTEMPT_NAME).write_text("{}")
    os.symlink("generations/gen-current", base / "current")
    return base


def test_classifier_excludes_refs_unknown_and_uncertain(root):
    store = root / "generations"
    _legacy(root, _id(1), candidate=str(store / "gen-old"))
    _legacy(root, _id(2), candidate=str(store / "gen-current"))
    _legacy(root, _id(3), phase="prepared", candidate=str(store / "gen-prepared"))
    _legacy(root, _id(4), candidate=str(store / "gen-attempt"))
    _legacy(root, _id(5), candidate="/somewhere/else")
    _legacy(root, _id(6), current=str(store / "gen-current"), candidate="")
    _legacy(root, _id(7), phase="failed", candidate="")
    (root / tx.OPERATION_DIR / "not-an-id.json").write_text("{}")
    (root / tx.OPERATION_DIR / f"{_id(8)}.json").write_text("{torn")
    _legacy(root, _id(9)).write_text(json.dumps({"operation_id": _id(10)}))
    namespace = root / tx.OPERATION_DIR
    os.symlink(namespace / f"{_id(1)}.json", namespace / f"{_id(11)}.json")
    before = _snapshot(root)

    manifest = rm.dry_run_manifest(root)

    assert _snapshot(root) == before, "dry run must not touch a single byte or mtime"
    assert manifest["mutation_performed"] is False and manifest["dry_run"] is True
    assert sorted(entry["operation_id"] for entry in manifest["candidates"]) == [_id(1), _id(7)]
    reasons = {row["relpath"].split("/")[-1]: row["reason"] for row in manifest["excluded"]}
    assert reasons[f"{_id(2)}.json"] == "current_or_installed_ref"
    assert reasons[f"{_id(3)}.json"] == "prepared_ref"
    assert reasons[f"{_id(4)}.json"] == "generation_attempt_or_v2_locator"
    assert reasons[f"{_id(5)}.json"] == "uncertain_candidate"
    assert reasons[f"{_id(6)}.json"] == "current_or_installed_ref"
    assert reasons["not-an-id.json"] == "unknown_namespace_entry"
    assert reasons[f"{_id(8)}.json"] == "unreadable_or_unsafe_record"
    assert reasons[f"{_id(9)}.json"] == "unreadable_or_unsafe_record"
    assert reasons[f"{_id(11)}.json"] == "unreadable_or_unsafe_record"
    assert len(manifest["candidates"]) + len(manifest["excluded"]) == 11


def test_candidate_label_never_claims_terminal(root):
    _legacy(root, _id(1), phase="partial")
    entry = rm.dry_run_manifest(root)["candidates"][0]
    assert entry["class"] == "legacy_abandonment_candidate"
    assert entry["finality"] == "unproven" and entry["protected"] is True
    assert "ABANDONMENT CANDIDATE" in entry["label"]
    # Paths are excluded: pytest names tmp directories after the test itself.
    text = json.dumps({k: v for k, v in entry.items() if k not in ("path", "relpath")})
    text = text.lower()
    assert "terminal" not in text and "retir" not in text and "complete" not in text


def test_manifest_entry_records_exact_facts_and_restore_transaction(root):
    path = _legacy(root, _id(1))
    os.chmod(path, 0o640)
    os.utime(path, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
    data = path.read_bytes()
    entry = rm.dry_run_manifest(root)["candidates"][0]
    assert entry["path"] == str(path)
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["bytes"] == len(data)
    assert entry["mtime_ns"] == 1_700_000_000_123_456_789
    assert entry["mode"] == 0o640
    restore = entry["restore"]
    assert restore["op"] == "restore_file" and restore["precondition"] == "absent_or_identical"
    assert restore["content_sha256"] == entry["sha256"] and restore["mode"] == 0o640


def test_root_with_unresolvable_pointer_excludes_everything(root):
    _legacy(root, _id(1))
    (root / "current").unlink()
    (root / "current").mkdir()
    manifest = rm.dry_run_manifest(root)
    assert manifest["candidates"] == []
    assert manifest["excluded"][0]["reason"].startswith("root_uncertain:")


def test_installed_pointer_other_than_current_is_a_ref(root):
    os.symlink("generations/gen-old/bin/python", root / "daemon-python")
    _legacy(root, _id(1), candidate=str(root / "generations" / "gen-old"))
    assert rm.dry_run_manifest(root)["candidates"] == []


def test_truncated_scan_is_explicit(root):
    for index in range(5):
        _legacy(root, _id(index + 1))
    manifest = rm.dry_run_manifest(root, max_entries=3)
    assert manifest["scan_complete"] is False and len(manifest["candidates"]) == 3


def test_v2_namespace_is_reported_out_of_scope(root):
    (root / ".update-operations-v2").mkdir()
    manifest = rm.dry_run_manifest(root)
    expected = {"relpath": ".update-operations-v2", "reason": "new_protocol_namespace_out_of_scope"}
    assert expected in manifest["excluded"]


def _approved(manifest: dict[str, Any]) -> str:
    """What a reviewer signs off: the digest of the manifest they read."""
    return manifest["manifest_sha256"]


def _resign(manifest: dict[str, Any]) -> dict[str, Any]:
    """Recompute the self-digest after an edit, as a forger would."""
    body = {key: manifest[key] for key in ("schema", "root", "candidates")}
    manifest["manifest_sha256"] = hashlib.sha256(rm._canonical(body)).hexdigest()
    return manifest


def _rel(index: int, namespace: str = tx.OPERATION_DIR) -> str:
    return f"{namespace}/{_id(index)}.json"


def test_matching_reviewed_manifest_round_trip_is_lossless(root, tmp_path):
    path = _legacy(root, _id(1), receipts=[{"unit": "serve", "state": "pending"}])
    os.chmod(path, 0o600)
    os.utime(path, ns=(1_600_000_000_000_000_001, 1_600_000_000_000_000_001))
    original = path.read_bytes()
    manifest = rm.dry_run_manifest(root)
    store = tmp_path / "store"
    before = _snapshot(root)
    captured = rm.capture_content(manifest, store, approved_sha256=_approved(manifest))
    assert captured["captured"] == [_rel(1)]
    assert _snapshot(root) == before, "capture never touches the source"

    # Simulate a later operator-approved archive elsewhere: the record is absent.
    restored_root = tmp_path / "restored"
    report = rm.restore_manifest(
        manifest, store, restored_root, approved_sha256=_approved(manifest)
    )
    target = restored_root / tx.OPERATION_DIR / f"{_id(1)}.json"
    assert report["restored"] == [_rel(1)]
    assert target.read_bytes() == original
    assert target.stat().st_mtime_ns == 1_600_000_000_000_000_001
    assert target.stat().st_mode & 0o7777 == 0o600
    assert [p.name for p in target.parent.iterdir() if p.name.endswith(".tmp")] == []

    again = rm.restore_manifest(manifest, store, restored_root, approved_sha256=_approved(manifest))
    assert again == {"restored": [], "already_present": [_rel(1)]}


def test_unreviewed_or_resigned_manifest_is_refused(root, tmp_path):
    _legacy(root, _id(1))
    manifest = rm.dry_run_manifest(root)
    reviewed = _approved(manifest)
    with pytest.raises(rm.ManifestError, match="reviewed manifest"):
        rm.capture_content(manifest, tmp_path / "store", approved_sha256="0" * 64)
    forged = _resign(json.loads(json.dumps(manifest)))
    forged["candidates"][0]["restore"]["mode"] = 0o777
    _resign(forged)
    with pytest.raises(rm.ManifestError, match="reviewed manifest"):
        rm.restore_manifest(forged, tmp_path / "store", tmp_path / "x", approved_sha256=reviewed)
    tampered = json.loads(json.dumps(manifest))
    tampered["candidates"][0]["restore"]["mode"] = 0o777
    with pytest.raises(rm.ManifestError, match="digest"):
        rm.restore_manifest(tampered, tmp_path / "store", tmp_path / "x", approved_sha256=reviewed)
    assert not (tmp_path / "store").exists() and not (tmp_path / "x").exists()


def test_hash_mismatch_on_disk_refuses_capture_before_any_write(root, tmp_path):
    _legacy(root, _id(1))
    path = _legacy(root, _id(2))
    manifest = rm.dry_run_manifest(root)
    path.write_text(json.dumps({"operation_id": _id(2), "phase": "promoted"}))
    with pytest.raises(rm.ManifestError, match="no longer match"):
        rm.capture_content(manifest, tmp_path / "store", approved_sha256=_approved(manifest))
    assert not (tmp_path / "store").exists(), "all-or-nothing: entry 1 was not captured either"


def test_hash_mismatch_in_store_refuses_restore_without_writing(root, tmp_path):
    _legacy(root, _id(1))
    manifest = rm.dry_run_manifest(root)
    store = tmp_path / "store"
    rm.capture_content(manifest, store, approved_sha256=_approved(manifest))
    digest = manifest["candidates"][0]["sha256"]
    os.chmod(store / digest, 0o600)
    (store / digest).write_text("{}")
    target_root = tmp_path / "restored"
    with pytest.raises(rm.ManifestError, match="do not match"):
        rm.restore_manifest(manifest, store, target_root, approved_sha256=_approved(manifest))
    assert not (target_root / tx.OPERATION_DIR).exists()


def test_restore_never_overwrites_a_different_record(root, tmp_path):
    _legacy(root, _id(1))
    manifest = rm.dry_run_manifest(root)
    store = tmp_path / "store"
    rm.capture_content(manifest, store, approved_sha256=_approved(manifest))
    path = root / tx.OPERATION_DIR / f"{_id(1)}.json"
    path.write_text(json.dumps({"operation_id": _id(1), "phase": "failed"}))
    changed = path.read_bytes()
    with pytest.raises(rm.ManifestError, match="different record"):
        rm.restore_manifest(manifest, store, root, approved_sha256=_approved(manifest))
    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    "relpath",
    [
        "../outside.json",
        f"{tx.OPERATION_DIR}/../{_id(1)}.json",
        f"generations/{_id(1)}.json",
        f"{tx.OPERATION_DIR}/nested/{_id(1)}.json",
        f"/{tx.OPERATION_DIR}/{_id(1)}.json",
        f"{tx.OPERATION_DIR}/{_id(1)}.txt",
    ],
)
def test_outside_namespace_path_refuses_even_when_reviewed(root, tmp_path, relpath):
    """A reviewer approving the digest still cannot approve an escaping path."""
    _legacy(root, _id(1))
    manifest = json.loads(json.dumps(rm.dry_run_manifest(root)))
    manifest["candidates"][0]["relpath"] = relpath
    manifest["candidates"][0]["restore"]["relpath"] = relpath
    _resign(manifest)
    with pytest.raises(rm.ManifestError, match="outside the operation namespaces"):
        rm.capture_content(manifest, tmp_path / "store", approved_sha256=_approved(manifest))
    with pytest.raises(rm.ManifestError, match="outside the operation namespaces"):
        rm.restore_manifest(
            manifest, tmp_path / "store", tmp_path / "x", approved_sha256=_approved(manifest)
        )
    assert not (tmp_path / "store").exists() and not (tmp_path / "x").exists()


def test_recorded_path_with_dotdot_escape_refuses_capture(root, tmp_path):
    _legacy(root, _id(1))
    outside = tmp_path / "outside"
    outside.mkdir()
    manifest = json.loads(json.dumps(rm.dry_run_manifest(root)))
    manifest["candidates"][0]["path"] = str(
        root / tx.OPERATION_DIR / ".." / ".." / "outside" / f"{_id(1)}.json"
    )
    _resign(manifest)
    with pytest.raises(rm.ManifestError, match="disagrees"):
        rm.capture_content(manifest, tmp_path / "store", approved_sha256=_approved(manifest))


def test_symlinked_namespace_escape_refuses_capture_and_restore(root, tmp_path):
    """Same bytes reachable through a symlinked namespace are still refused."""
    path = _legacy(root, _id(1))
    manifest = rm.dry_run_manifest(root)
    reviewed = _approved(manifest)
    outside = tmp_path / "outside-ns"
    outside.mkdir(mode=0o700)
    (outside / path.name).write_bytes(path.read_bytes())
    os.chmod(outside / path.name, 0o600)
    moved = tmp_path / "moved-ns"
    os.rename(root / tx.OPERATION_DIR, moved)
    os.symlink(outside, root / tx.OPERATION_DIR)
    with pytest.raises(rm.ManifestError, match="escapes the install root"):
        rm.capture_content(manifest, tmp_path / "store", approved_sha256=reviewed)
    assert not (tmp_path / "store").exists()

    # Restore side: a real store, a target root whose namespace is a symlink.
    os.unlink(root / tx.OPERATION_DIR)
    os.rename(moved, root / tx.OPERATION_DIR)
    store = tmp_path / "store"
    rm.capture_content(manifest, store, approved_sha256=reviewed)
    target_root = tmp_path / "restored"
    target_root.mkdir()
    escape = tmp_path / "escape"
    escape.mkdir(mode=0o700)
    os.symlink(escape, target_root / tx.OPERATION_DIR)
    with pytest.raises(rm.ManifestError, match="escapes the install root"):
        rm.restore_manifest(manifest, store, target_root, approved_sha256=reviewed)
    assert list(escape.iterdir()) == []


def test_symlinked_target_file_refuses_restore(root, tmp_path):
    _legacy(root, _id(1))
    manifest = rm.dry_run_manifest(root)
    store = tmp_path / "store"
    rm.capture_content(manifest, store, approved_sha256=_approved(manifest))
    target_root = tmp_path / "restored"
    (target_root / tx.OPERATION_DIR).mkdir(parents=True, mode=0o700)
    victim = tmp_path / "victim.json"
    os.symlink(victim, target_root / tx.OPERATION_DIR / f"{_id(1)}.json")
    with pytest.raises(rm.ManifestError, match="escapes its operation directory"):
        rm.restore_manifest(manifest, store, target_root, approved_sha256=_approved(manifest))
    assert not victim.exists()


def test_v2_namespace_target_is_accepted_by_the_path_check(root, tmp_path):
    """Both operation namespaces are inside the boundary; nothing else is."""
    assert rm._relpath_ok(_rel(1, tx.V2_OPERATION_DIR))
    assert rm._relpath_ok(_rel(1))
    assert not rm._relpath_ok(f".update-operations-v3/{_id(1)}.json")


def test_capture_refuses_in_root_store(root, tmp_path):
    _legacy(root, _id(1))
    manifest = rm.dry_run_manifest(root)
    with pytest.raises(rm.ManifestError, match="outside"):
        rm.capture_content(manifest, root / "store", approved_sha256=_approved(manifest))
    assert not (root / "store").exists()
