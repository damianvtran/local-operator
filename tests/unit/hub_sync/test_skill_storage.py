"""Narrow journal/owned-reader guards; not engine or remote publication coverage."""

from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.config_writer import nonblocking_file_claim
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.hub_sync.provenance import make_record, read_baseline
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import (
    ManagedSkillStorage,
    managed_skill_read,
)
from local_operator.hub_sync.transactions import LocalTransaction, TransactionError
from local_operator.skills.discovery import discover_skills, managed_root_context
from local_operator.skills.protocol import resolve_skill_url

from .test_skill_document import manifest


def _input(body: str = "# Body\n"):
    value = manifest(
        files=[{"path": "scripts/inert.py", "content": "raise RuntimeError('never run')\n"}]
    )
    value["files"][0]["content"] = value["files"][0]["content"].replace("# Body\n", body)
    document = SkillDocument.from_json(value)
    identity = LinkIdentity(
        "skill", "immutable-local", "https://example.invalid/v1", "public", None, "remote"
    )
    baseline = replace(
        make_record("skill", identity.local_id, identity.hub_id, None, document.to_json(), "pull"),
        identity=identity,
        remote_revision='"revision-1"',
        lineage="clean",
        local_alias=document.name,
    )
    return identity, document, baseline


def test_absent_native_discovery_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "absent-profile"
    with managed_skill_read(root):
        assert not root.exists()
    assert discover_skills([root / "skills"], config_dir=root) == ([], [])
    assert not root.exists()


def test_world_scan_cannot_recover_external_profile(tmp_path: Path) -> None:
    root = tmp_path / "outside-profile"
    world = tmp_path / "world"
    world.mkdir()
    with managed_root_context(root / "skills", root, boundary=world):
        assert not root.exists()
    assert not root.exists()


def test_kernel_claim_is_nonreentrant_and_released(tmp_path: Path) -> None:
    path = tmp_path / "operation.lock"
    with nonblocking_file_claim(path) as first:
        assert first
        with nonblocking_file_claim(path) as second:
            assert not second
    with nonblocking_file_claim(path) as following:
        assert following


def test_malformed_journal_fingerprint_cannot_supply_an_absence_basis() -> None:
    identity, _, _ = _input()
    with pytest.raises(TransactionError, match="fingerprint"):
        LocalTransaction(
            operation_id="a" * 32,
            kind="skill",
            local_id=identity.local_id,
            link=identity.to_json(),
            direction="pull",
            before_fingerprint=False,
            after_fingerprint=None,
            payload={},
        )


def test_managed_commit_is_complete_inert_and_protocol_readable(tmp_path: Path) -> None:
    identity, document, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    transaction = storage.apply(
        identity, document.name, document, baseline, expected_fingerprint=None
    )
    assert transaction.phase == "complete"
    assert storage.snapshot(identity.local_id).document == document
    assert read_baseline(storage.config_dir, "skill", identity.local_id).identity == identity
    script = storage.path(document.name) / "scripts/inert.py"
    assert script.read_text() == "raise RuntimeError('never run')\n"
    assert script.stat().st_mode & 0o111 == 0
    resources, _ = discover_skills([storage.root], config_dir=storage.config_dir)
    resource = next(row for row in resources if row.name == document.name)
    assert resource.managed_config_dir == storage.config_dir
    assert "# Body" in resolve_skill_url("skill://example-skill", {resource.name: resource})


def test_crash_gap_restores_only_verified_original_backup(tmp_path: Path) -> None:
    identity, original, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(identity, original.name, original, baseline, expected_fingerprint=None)
    _, changed, next_baseline = _input("# New body\n")

    def crash(phase: str) -> None:
        if phase == "old-renamed":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        storage.apply(
            identity,
            changed.name,
            changed,
            next_baseline,
            expected_fingerprint=original.fingerprint,
            checkpoint=crash,
        )
    storage.recover()
    assert storage.snapshot(identity.local_id).document == original


def test_crash_gap_preserves_external_backup_edit_instead_of_promoting_it(tmp_path: Path) -> None:
    identity, original, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(identity, original.name, original, baseline, expected_fingerprint=None)
    _, changed, next_baseline = _input("# New body\n")

    def crash(phase: str) -> None:
        if phase == "old-renamed":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        storage.apply(
            identity,
            changed.name,
            changed,
            next_baseline,
            expected_fingerprint=original.fingerprint,
            checkpoint=crash,
        )
    pending = storage.journal.pending(kind="skill")[0]
    destination, _, backup = storage._locations(pending)
    body = backup / "SKILL.md"
    body.write_text(body.read_text() + "\nExternal authored edit.\n")
    actual = body.read_bytes()
    storage.recover()
    assert not destination.exists()
    assert body.read_bytes() == actual
    assert storage.journal.read(pending.operation_id).phase == "blocked"
