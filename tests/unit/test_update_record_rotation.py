"""Recovery-record rotation: the store retires finished records instead of filling up.

Operator request (2026-11-16): "make sure that this store properly rotates so that
users don't run into this hard limit". A Mac hit the 256-entry cap in
``.update-operations`` and needed a reviewed manual archive. These tests pin the
automatic replacement: sealed v2 records rotate into an in-root archive, legacy
candidates of the reviewed classifier are migrated under pressure, protected records
never move, the archive is bounded, and the typed 256 refusal still fires when only
protected records remain. Clocks are injected; nothing sleeps for aging.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator import fenced_operation as v2
from local_operator import install_transaction as tx
from local_operator import recovery_manifest as manifest_mod

_SENTENCE = "update recovery records are at capacity; inspect pending operations"
_T0 = 1_800_000_000.0
_DAY = 24 * 60 * 60


def _root(tmp_path: Path) -> Path:
    root = tx.canonical_root(tmp_path / "root")
    (root / "generations").mkdir(parents=True)
    return root


def _sealed_record(identifier: str, sealed_at: float, **overrides: Any) -> dict[str, Any]:
    """A v2 record in exactly the shape ``finish(..., seal=True)`` leaves behind."""
    slot = "a" * 32
    record: dict[str, Any] = {
        "protocol": tx.V2_PROTOCOL,
        "epoch": v2.EPOCH,
        "operation_id": identifier,
        "baseline": "A",
        "target": "B",
        "route": "default-cli",
        "phase": "partial",
        "candidate": "",
        "receipts": [],
        "enrollment": "closed",
        "enrollment_closed_by": slot,
        "writers": {slot: {"role": "coordinator", "epoch": 1, "state": "finished"}},
        "obligations": {},
        "occurrences": {},
        "continuations": {},
        "linked_generations": [],
        "sealed": {"sealed_at": sealed_at, "sealed_by": slot},
        "created_at": sealed_at,
        "updated_at": sealed_at,
    }
    record.update(overrides)
    return record


def _write_v2(root: Path, record: dict[str, Any]) -> Path:
    directory = root / tx.V2_OPERATION_DIR
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / f"{record['operation_id']}.json"
    path.write_text(json.dumps(record, sort_keys=True))
    path.chmod(0o600)
    return path


def _fill_sealed(root: Path, count: int, base: float = 1000.0) -> list[str]:
    ids = []
    for index in range(count):
        identifier = uuid.uuid4().hex
        _write_v2(root, _sealed_record(identifier, base + index))
        ids.append(identifier)
    return ids


def _write_legacy(root: Path, phase: str = "partial", candidate: str = "", age: float = 2 * _DAY):
    """A legacy record shaped like the incident's, with an mtime ``age`` seconds old."""
    directory = root / tx.OPERATION_DIR
    directory.mkdir(mode=0o700, exist_ok=True)
    identifier = uuid.uuid4().hex
    path = directory / f"{identifier}.json"
    record = {
        "operation_id": identifier,
        "phase": phase,
        "candidate": candidate,
        "baseline": "0.1.0",
        "target": "0.1.1",
        "route": "latest",
        "receipts": [],
        "updated_at": 0,
    }
    path.write_text(json.dumps(record, sort_keys=True))
    path.chmod(0o640 if phase == "failed" else 0o600)
    stamp = int((_now() - age) * 1e9)
    os.utime(path, ns=(stamp, stamp))
    return path


def _now() -> float:
    return time.time()


def _fingerprint(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        stat.S_IMODE(info.st_mode),
        (info.st_mtime_ns),
    )


def _rotate(root: Path, now: float) -> dict[str, list[str]]:
    with tx.transaction(root, operation="test rotation") as held:
        return tx.rotate_for_admission(root, held, now=now)


def _live(root: Path, namespace: str) -> set[str]:
    directory = root / namespace
    return {path.name for path in directory.iterdir()} if directory.exists() else set()


def _archived(root: Path) -> list[str]:
    directory = root / tx.ARCHIVE_DIR
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def _tree(path: Path) -> dict[str, tuple[str, int, int] | str]:
    out: dict[str, tuple[str, int, int] | str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            out[str(item)] = "->" + os.readlink(item)
        elif item.is_file():
            out[str(item)] = _fingerprint(item)
        else:
            out[str(item)] = "dir"
    return out


def test_300_sealed_v2_rotate_to_keep_and_archive_is_bounded_by_count_and_age(tmp_path):
    root = _root(tmp_path)
    ids = _fill_sealed(root, 300)
    report = _rotate(root, _T0)
    live = _live(root, tx.V2_OPERATION_DIR)
    assert len(live) == tx.ROTATION_KEEP_SEALED == 32
    # Oldest seals go first: exactly the newest 32 stay live.
    assert live == {f"{identifier}.json" for identifier in ids[-32:]}
    assert len(report["v2_archived"]) == 268 == len(_archived(root))
    assert tx.active_recovery_entries(root) == 32
    # A foreign file in the archive is never indexed, aged or deleted.
    foreign = root / tx.ARCHIVE_DIR / "operator-notes.txt"
    foreign.write_text("keep")

    # Count bound: another 300 push the archive past 512; the OLDEST stamps go.
    second = _fill_sealed(root, 300, base=5000.0)
    _rotate(root, _T0 + 10)
    names = [name for name in _archived(root) if name != foreign.name]
    assert len(names) == tx.ARCHIVE_MAX_ENTRIES == 512
    stamps = {name.split(".")[0] for name in names}
    assert stamps == {f"{int(_T0 * 1000):013d}", f"{int((_T0 + 10) * 1000):013d}"}
    newer = [name for name in names if name.startswith(f"{int((_T0 + 10) * 1000):013d}")]
    assert len(newer) == 300 - 32 + 32  # the 32 previously-live + 268 of the second batch
    assert {f"{identifier}.json" for identifier in second[-32:]} == _live(root, tx.V2_OPERATION_DIR)

    # Age bound with an injected clock: just under 30 days keeps everything...
    assert _rotate(root, _T0 + tx.ARCHIVE_MAX_AGE_S)["aged"] == []
    # ...and just past it drops only the first batch's stamp, oldest first.
    aged = _rotate(root, _T0 + tx.ARCHIVE_MAX_AGE_S + 5)["aged"]
    assert aged and all(name.startswith(f"{int(_T0 * 1000):013d}") for name in aged)
    assert len(_archived(root)) == 300 + 1
    assert foreign.read_text() == "keep"


def test_protected_records_are_never_moved(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "ROTATION_KEEP_SEALED", 0)
    root = _root(tmp_path)
    generations = root / "generations"
    for name in ("current-gen", "previous-gen", "attempted-gen"):
        (generations / name).mkdir()
    (root / "current").symlink_to(generations / "current-gen")
    (root / "previous").symlink_to(generations / "previous-gen")
    (generations / "attempted-gen" / tx.ATTEMPT_NAME).write_text("{}")
    protected = [
        _write_v2(
            root, _sealed_record(uuid.uuid4().hex, 1, candidate=str(generations / "current-gen"))
        ),
        _write_v2(
            root,
            _sealed_record(
                uuid.uuid4().hex, 2, linked_generations=[str(generations / "previous-gen")]
            ),
        ),
        _write_v2(root, _sealed_record(uuid.uuid4().hex, 3, phase="prepared")),
        _write_v2(root, _sealed_record(uuid.uuid4().hex, 4, enrollment="open")),
        _write_v2(
            root,
            _sealed_record(
                uuid.uuid4().hex,
                5,
                writers={"b" * 32: {"role": "reporter", "epoch": 1, "state": "open"}},
            ),
        ),
        _write_v2(root, _sealed_record(uuid.uuid4().hex, 6, candidate="/elsewhere/gen")),
        _write_v2(
            root,
            _sealed_record(uuid.uuid4().hex, 7, candidate=str(generations / "attempted-gen")),
        ),
        _write_v2(root, _sealed_record(uuid.uuid4().hex, 8, sealed=None)),
    ]
    # A real open v2 operation with an enrolled writer, through the API.
    context = v2.start_operation(root, baseline="A", target="B", route="default-cli")
    v2.enroll_writer(context, "reporter")
    protected.append(root / tx.V2_OPERATION_DIR / f"{context.operation_id}.json")
    legacy_protected = [
        _write_legacy(root, phase="prepared"),
        _write_legacy(root, candidate=str(generations / "current-gen")),
        _write_legacy(root, candidate=str(generations / "attempted-gen")),
        _write_legacy(root, age=60),  # changed within the grace window
    ]
    retirable = _write_v2(root, _sealed_record(uuid.uuid4().hex, 9))
    before = {path: _fingerprint(path) for path in protected + legacy_protected}
    # Push legacy+v2 past the migration threshold so the legacy path runs too.
    filler = [_write_legacy(root, phase="queued") for _ in range(tx.LEGACY_MIGRATION_THRESHOLD)]
    report = _rotate(root, _now())
    assert report["v2_archived"] == [retirable.stem]
    assert report["legacy_archived"] == []
    assert {path: _fingerprint(path) for path in protected + legacy_protected} == before
    assert all(path.exists() for path in filler)


def test_legacy_store_at_256_is_archived_and_admission_succeeds(tmp_path):
    root = _root(tmp_path)
    (root / "generations" / "live").mkdir()
    (root / "current").symlink_to(root / "generations" / "live")
    excluded = [
        _write_legacy(root, phase="prepared"),
        _write_legacy(root, candidate=str(root / "generations" / "live")),
    ]
    candidates = [
        _write_legacy(root, phase="failed" if index % 7 == 0 else "partial") for index in range(254)
    ]
    assert tx.active_recovery_entries(root) == 256
    prints = {path.stem: _fingerprint(path) for path in candidates}
    operation = tx.start_operation(root, baseline="A", target="B", route="latest")
    # The two excluded records and the new operation are all that stay live.
    assert _live(root, tx.OPERATION_DIR) == {
        *(path.name for path in excluded),
        f"{operation.identifier}.json",
    }
    archived = {name.split(".")[2]: root / tx.ARCHIVE_DIR / name for name in _archived(root)}
    assert set(archived) == set(prints)
    # Rename keeps the inode: bytes, mode (incl. a non-default 0o640) and mtime.
    assert {identifier: _fingerprint(path) for identifier, path in archived.items()} == prints

    # Restore is a rename back, byte-for-byte, and refuses an unknown id.
    identifier = candidates[0].stem
    restored = tx.restore_archived(root, identifier)
    assert restored == candidates[0]
    assert _fingerprint(restored) == prints[identifier]
    assert tx.operation_status(root, identifier)["operation_id"] == identifier
    with pytest.raises(tx.InstallTransactionError, match="no archived"):
        tx.restore_archived(root, identifier)


def test_only_protected_records_at_256_still_refuse_with_the_typed_error(tmp_path):
    root = _root(tmp_path)
    for _ in range(256):
        _write_legacy(root, phase="prepared")
    before = _tree(root)
    for admit in (tx.start_operation, v2.start_operation):
        with pytest.raises(tx.RecoveryCapacityError) as caught:
            admit(root, baseline="A", target="B", route="latest")
        assert str(caught.value) == _SENTENCE
        assert caught.value.code == "update_recovery_capacity"
    assert {k: v for k, v in _tree(root).items() if not k.endswith(tx.LOCK_NAME)} == before


def test_negative_control_rotation_disabled_refuses_at_256(tmp_path, monkeypatch):
    """Same store the migration test unblocks; with rotation off it must still refuse."""
    monkeypatch.setattr(tx, "RECORD_ROTATION_ENABLED", False)
    root = _root(tmp_path)
    for _ in range(200):
        _write_legacy(root)
    _fill_sealed(root, 56)
    with pytest.raises(tx.RecoveryCapacityError) as caught:
        tx.start_operation(root, baseline="A", target="B", route="latest")
    assert str(caught.value) == _SENTENCE and caught.value.code == "update_recovery_capacity"
    assert not (root / tx.ARCHIVE_DIR).exists()
    assert tx.active_recovery_entries(root) == 256


def test_crash_midway_is_safe_and_resumable(tmp_path, monkeypatch):
    root = _root(tmp_path)
    paths = [_write_legacy(root) for _ in range(256)]
    prints = {path.stem: _fingerprint(path) for path in paths}
    real_move = tx._move_entry
    calls = {"n": 0}

    def crash_after_ten(source: Path, target: Path) -> None:
        if calls["n"] == 10:
            raise OSError("simulated crash")
        calls["n"] += 1
        real_move(source, target)

    monkeypatch.setattr(tx, "_move_entry", crash_after_ten)
    with tx.transaction(root, operation="test") as held:
        assert held
        manifest = manifest_mod.dry_run_manifest(root)
        with pytest.raises(OSError, match="simulated"):
            manifest_mod.archive_candidates(
                manifest, approved_sha256=manifest["manifest_sha256"], now=_now()
            )
    # Nothing lost: every record is in exactly one place, bytes intact.
    archived = {name.split(".")[2]: root / tx.ARCHIVE_DIR / name for name in _archived(root)}
    live = {path.stem: path for path in (root / tx.OPERATION_DIR).iterdir()}
    assert len(archived) == 10 and len(live) == 246 and not set(archived) & set(live)
    assert {k: _fingerprint(v) for k, v in {**archived, **live}.items()} == prints

    # Back at the cap with protected (prepared) records, an admission during the
    # outage reports the SAME typed refusal, chained to the rotation failure.
    for _ in range(10):
        _write_legacy(root, phase="prepared")
    with pytest.raises(tx.RecoveryCapacityError) as caught:
        calls["n"] = 10
        tx.start_operation(root, baseline="A", target="B", route="latest")
    assert str(caught.value) == _SENTENCE
    assert isinstance(caught.value.__cause__, OSError)

    # Rerun with the SAME manifest: the 10 moved ones are recognised, not errors.
    monkeypatch.setattr(tx, "_move_entry", real_move)
    with tx.transaction(root, operation="test") as held:
        assert held
        result = manifest_mod.archive_candidates(
            manifest, approved_sha256=manifest["manifest_sha256"], now=_now()
        )
    assert len(result["already_archived"]) == 10 and len(result["archived"]) == 246
    archived = {name.split(".")[2]: root / tx.ARCHIVE_DIR / name for name in _archived(root)}
    assert {k: _fingerprint(v) for k, v in archived.items()} == prints
    assert tx.active_recovery_entries(root) == 10  # only the protected prepared ones


def test_a_tampered_candidate_refuses_before_the_first_move(tmp_path):
    root = _root(tmp_path)
    paths = [_write_legacy(root) for _ in range(5)]
    with tx.transaction(root, operation="test"):
        manifest = manifest_mod.dry_run_manifest(root)
        paths[-1].write_text(paths[-1].read_text() + " ")
        with pytest.raises(manifest_mod.ManifestError, match="no longer match"):
            manifest_mod.archive_candidates(
                manifest, approved_sha256=manifest["manifest_sha256"], now=_now()
            )
    assert all(path.exists() for path in paths)
    assert not (root / tx.ARCHIVE_DIR).exists()


def test_two_updaters_are_serialised_by_the_root_lock(tmp_path):
    root = _root(tmp_path)
    _fill_sealed(root, 300)
    finished = threading.Event()
    errors: list[BaseException] = []

    def second_updater() -> None:
        try:
            tx.start_operation(root, baseline="A", target="B", route="latest")
        except BaseException as exc:  # surfaced below, never swallowed
            errors.append(exc)
        finished.set()

    with tx.transaction(root, operation="first updater"):
        # Another PROCESS cannot take the root while it is held: LockBusy.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path\n"
                "from local_operator import install_transaction as tx\n"
                "try:\n"
                "    with tx.transaction(Path(sys.argv[1]), operation='p', timeout=0.2):\n"
                "        sys.exit(3)\n"
                "except tx.LockBusy:\n"
                "    sys.exit(0)\n",
                str(root),
            ],
            capture_output=True,
            timeout=60,
        )
        assert probe.returncode == 0, probe.stderr
        thread = threading.Thread(target=second_updater)
        thread.start()
        # Another THREAD waits on the same root rather than rotating beside us.
        assert not finished.wait(0.3)
        assert len(_live(root, tx.V2_OPERATION_DIR)) == 300
    thread.join(30)
    assert finished.is_set() and not errors

    workers = [threading.Thread(target=second_updater) for _ in range(4)]
    finished.clear()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
    assert not errors
    live_v2 = _live(root, tx.V2_OPERATION_DIR)
    archived = _archived(root)
    # No record lost or duplicated across the concurrent rotations.
    assert len(live_v2) == 32 and len(archived) == 268
    assert len(set(name.split(".")[2] for name in archived) | {n[:-5] for n in live_v2}) == 300
    assert len(_live(root, tx.OPERATION_DIR)) == 5


def test_observers_never_rotate_and_can_read_an_archived_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "ROTATION_KEEP_SEALED", 0)
    root = _root(tmp_path)
    generation = root / "generations" / "old"
    generation.mkdir()
    context = v2.start_operation(root, baseline="A", target="B", route="default-cli")
    v2.link_generation(context, generation)
    # Past the prepared phases: a sealed record still at "queued" stays live.
    v2.advance(context, "partial")
    assert v2.finish(context, "completed", seal=True)["sealed"] is True
    _fill_sealed(root, 300)
    for _ in range(10):
        _write_legacy(root)
    before = _tree(root)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an observer rotated, moved or aged a record")

    with monkeypatch.context() as patched:
        for name in ("rotate_for_admission", "_archive_moves", "_age_archive", "_move_entry"):
            patched.setattr(tx, name, forbidden)
        assert v2.observe_generation(generation)["sealed"]
        assert v2.operation_status(root, context.operation_id)["writer"] is None
        tx.recovery_capacity_status(root)
        tx.active_recovery_entries(root)
    assert _tree(root) == before

    # After admission rotates it away, the observer still finds the finished seal.
    _rotate(root, _now())
    assert not (root / tx.V2_OPERATION_DIR / f"{context.operation_id}.json").exists()
    status = v2.observe_generation(generation)
    assert status["archived"] is True and status["writer"] is None and status["sealed"]


def test_rotation_and_aging_never_touch_a_sibling_archive_outside_the_root(tmp_path):
    root = _root(tmp_path)
    sibling = tmp_path / "lop-record-archive" / "20261116"
    sibling.mkdir(parents=True)
    old = int((_T0 - 400 * _DAY) * 1e9)
    for index in range(3):
        name = f"{int(_T0 * 1000) - index:013d}.v1.{uuid.uuid4().hex}.json"
        (sibling / name).write_text(f"record {index}")
        os.utime(sibling / name, ns=(old, old))
    (tmp_path / "lop-record-archive" / "store").mkdir()
    before = _tree(tmp_path / "lop-record-archive")
    _fill_sealed(root, 100)
    _rotate(root, _T0)
    aged = _rotate(root, _T0 + 365 * _DAY)["aged"]
    assert len(aged) == 100 - 32
    assert _tree(tmp_path / "lop-record-archive") == before
