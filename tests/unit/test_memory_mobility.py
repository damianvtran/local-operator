"""Portable truth is complete and atomic, never a partially imported chunk."""

import shutil

import pytest

from local_operator.memory import MemoryIdentity, MemoryStore, SESSION_EXPORT_NAME


def test_portable_import_validates_full_snapshot_before_one_export(tmp_path, monkeypatch):
    source_dir = tmp_path / "source" / "sessions" / "same-session"
    source = MemoryStore(tmp_path / "source", MemoryIdentity("same-session"), source_dir, origin="source-device")
    for n in range(140):
        source.put("session", {"title": f"Evidence {n}", "body": f"Evidence number {n} docs/proof.md"})
    target_dir = tmp_path / "target" / "sessions" / "same-session"
    target_dir.mkdir(parents=True)
    target_file = target_dir / SESSION_EXPORT_NAME
    shutil.copyfile(source_dir / SESSION_EXPORT_NAME, target_file)
    before = target_file.read_bytes()
    target = MemoryStore(tmp_path / "target", MemoryIdentity("same-session"), target_dir)
    original = target._export
    observed = []
    def stop_after_replace(db, owner, directory=None):
        observed.append(db.execute("SELECT count(*) FROM ops").fetchone()[0])
        original(db, owner)
        raise OSError("Synthetic interruption after snapshot replace, before index commit")
    monkeypatch.setattr(target, "_export", stop_after_replace)
    with pytest.raises(OSError):
        target.import_session()
    assert observed == [140], "Old chunked import would expose only 128 rows here"
    assert target_file.read_bytes() == before
    with target.connection() as db:
        assert db.execute("SELECT count(*) FROM ops").fetchone()[0] == 0
    monkeypatch.setattr(target, "_export", original)
    target.import_session()
    assert len(target_file.read_text().splitlines()) == 140
    with target.connection() as db:
        assert db.execute("SELECT count(*) FROM ops").fetchone()[0] == 140
    target.import_session()
    assert target_file.read_bytes() == before


def test_invalid_portable_file_or_scope_creates_no_bank(tmp_path):
    directory = tmp_path / "sessions" / "same-session"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("not portable memory")
    (directory / SESSION_EXPORT_NAME).symlink_to(outside)
    store = MemoryStore(tmp_path, MemoryIdentity("same-session"), directory)
    with pytest.raises(ValueError, match="regular file"):
        store.import_session()
    assert not store.path.exists()
    assert outside.read_text() == "not portable memory"
