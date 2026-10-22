"""Bundle form across storage, adapters and provenance classification.

The core carrier is exercised in ``test_skill_bundle``; this file pins the
INTEGRATION contract this slice adds: a bundle link is classified by its tree
hash, commits through its own journalled storage, recovers from a crash in the
two-rename gap, and leaves every document path byte-identical (the document
fingerprint guard is a regression check against those semantics moving).

Recovery tests craft the crashed state directly (journal + directories) because
a real crash lands BETWEEN the two renames: ``apply``'s own checkpoints fire
before and after the pair, never inside it.
"""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.adapters import SkillAdapter
from local_operator.hub_sync.identity import LinkError, LinkIdentity
from local_operator.hub_sync.skill_bundle import (
    SkillBundleManifest,
    build_bundle,
    extract_archive,
    tree_hash_of,
)
from local_operator.hub_sync.skill_bundle_storage import ManagedBundleStorage
from local_operator.hub_sync.skill_document import (
    SkillDocument,
    skill_manifest_from_json,
    skill_manifest_from_remote,
)
from local_operator.hub_sync.skill_storage import ManagedSkillStorage, managed_skill_read


def _identity(local_id: str = "immutable-local", hub_id: str = "remote-bundle-1") -> LinkIdentity:
    return LinkIdentity("skill", local_id, "https://example.invalid/v1", "public", None, hub_id)


def _source(tmp_path: Path, body: str = "# Body\n") -> Path:
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        "---\nname: pdf-forms\ndescription: Fill and validate PDF forms.\n---\n\n" + body,
        encoding="utf-8",
    )
    (root / "references").mkdir(exist_ok=True)
    (root / "references" / "guide.md").write_text("alpha\n" * 20, encoding="utf-8")
    return root


def _bundle(tmp_path: Path, body: str = "# Body\n") -> tuple[Path, SkillBundleManifest]:
    built = build_bundle(_source(tmp_path, body))
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(built.data)
    return archive, built.manifest


def _baseline(identity: LinkIdentity, manifest: SkillBundleManifest):
    return replace(
        prov.make_record(
            "skill", identity.local_id, identity.hub_id, None, manifest.to_json(), "pull"
        ),
        identity=identity,
        remote_revision='"revision-1"',
        lineage="clean",
        local_alias=manifest.name,
    )


def _payload(
    alias: str,
    baseline,
    manifest: SkillBundleManifest,
    *,
    before: str | None,
    after: str | None,
) -> dict:
    """The bundle adapter's journal payload (design §5 step 1)."""

    return {
        "adapter": ManagedBundleStorage.ADAPTER,
        "local_alias": alias,
        "baseline": baseline.to_json(),
        "policy_revision": "",
        "automatic": False,
        "archive_hash": manifest.bundle.hash,
        "before_tree": before,
        "after_tree": after,
    }


def _roots(root: Path) -> tuple[Path, Path]:
    return root / "skills", root / "hub" / "backups"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_apply_installs_the_tree_and_finalizes_the_baseline(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    baseline = _baseline(identity, manifest)
    storage = ManagedBundleStorage(root)

    transaction = storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        baseline,
        expected_fingerprint=None,
    )

    destination = storage.path(manifest.name)
    assert tree_hash_of(destination) == manifest.fingerprint
    for path in destination.rglob("*"):
        assert _mode(path) == (0o600 if path.is_file() else 0o700)
    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert stored is not None
    assert stored.fields == manifest.to_json()
    assert stored.fingerprint == manifest.bundle.tree_hash
    assert storage.journal.read(transaction.operation_id).phase == "complete"
    skills, _backups = _roots(root)
    assert not (skills / f".stage-{transaction.operation_id}").exists()


def test_apply_refuses_a_local_change_and_leaves_it_untouched(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        _baseline(identity, manifest),
        expected_fingerprint=None,
    )
    destination = storage.path(manifest.name)
    edited = destination / "references" / "guide.md"
    edited.write_text("locally edited\n", encoding="utf-8")
    edited_bytes = edited.read_bytes()

    with pytest.raises(LinkError) as caught:
        storage.apply(
            identity,
            manifest.name,
            archive,
            manifest,
            _baseline(identity, manifest),
            expected_fingerprint=manifest.fingerprint,
        )

    assert caught.value.code == "local-concurrent-edit"
    assert edited.read_bytes() == edited_bytes


def test_apply_refuses_an_alias_rename_for_bundles(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)

    with pytest.raises(LinkError) as caught:
        storage.apply(
            identity,
            "other-alias",
            archive,
            manifest,
            _baseline(identity, manifest),
            expected_fingerprint=None,
        )

    assert caught.value.code == "bundle-alias-rename-unsupported"
    assert not storage.path("other-alias").exists()


def test_an_update_replaces_the_tree_and_keeps_the_old_one_as_backup(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    first_archive, first = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        first.name,
        first_archive,
        first,
        _baseline(identity, first),
        expected_fingerprint=None,
    )
    second_archive = tmp_path / "second.zip"
    built = build_bundle(_source(tmp_path, "# New body\n"))
    second_archive.write_bytes(built.data)
    second = built.manifest
    assert second.bundle.tree_hash != first.bundle.tree_hash

    transaction = storage.apply(
        identity,
        second.name,
        second_archive,
        second,
        _baseline(identity, second),
        expected_fingerprint=first.fingerprint,
    )

    destination = storage.path(second.name)
    assert tree_hash_of(destination) == second.fingerprint
    _skills, backups = _roots(root)
    backup = backups / f"skill-{identity.local_id}-{transaction.operation_id}"
    assert tree_hash_of(backup) == first.fingerprint
    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert stored is not None and stored.fingerprint == second.fingerprint


def _crash_between_renames(
    storage: ManagedBundleStorage,
    identity: LinkIdentity,
    *,
    before: str | None,
    archive: Path,
    manifest: SkillBundleManifest,
) -> str:
    """The durable state a kill -9 leaves after the first rename, before the second."""

    baseline = _baseline(identity, manifest)
    transaction = storage.journal.prepare(
        identity,
        "pull",
        before_fingerprint=before,
        after_fingerprint=manifest.fingerprint,
        payload=_payload(
            manifest.name,
            baseline,
            manifest,
            before=before,
            after=manifest.fingerprint,
        ),
    )
    stage_root = storage.root / f".stage-{transaction.operation_id}"
    stage_root.mkdir(parents=True, mode=0o700)
    extract_archive(archive, stage_root / "content", expected=manifest.bundle)
    # First rename done: the old tree is the backup, the alias is now absent.
    os.replace(
        storage.path(manifest.name),
        storage.config_dir
        / "hub"
        / "backups"
        / f"skill-{identity.local_id}-{transaction.operation_id}",
    )
    return transaction.operation_id


def test_recovery_restores_the_backup_when_the_second_rename_never_happened(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        _baseline(identity, manifest),
        expected_fingerprint=None,
    )
    newer_archive, newer = _bundle(tmp_path / "next", "# New body\n")
    operation = _crash_between_renames(
        storage, identity, before=manifest.fingerprint, archive=newer_archive, manifest=newer
    )

    storage.recover()

    destination = storage.path(manifest.name)
    assert tree_hash_of(destination) == manifest.fingerprint
    assert not (storage.root / f".stage-{operation}").exists()
    assert storage.journal.read(operation).phase == "aborted"


def test_recovery_finalizes_when_the_new_tree_is_already_in_place(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        _baseline(identity, manifest),
        expected_fingerprint=None,
    )
    newer_archive, newer = _bundle(tmp_path / "next", "# New body\n")
    operation = _crash_between_renames(
        storage, identity, before=manifest.fingerprint, archive=newer_archive, manifest=newer
    )
    # Second rename done too: the alias holds the new tree, the stage is empty.
    os.replace(
        storage.root / f".stage-{operation}" / "content",
        storage.path(newer.name),
    )

    storage.recover()

    destination = storage.path(newer.name)
    assert tree_hash_of(destination) == newer.fingerprint
    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert stored is not None and stored.fields == newer.to_json()
    assert storage.journal.read(operation).phase == "complete"


def test_recovery_blocks_rather_than_overwriting_an_unexpected_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        _baseline(identity, manifest),
        expected_fingerprint=None,
    )
    newer_archive, newer = _bundle(tmp_path / "next", "# New body\n")
    operation = _crash_between_renames(
        storage, identity, before=manifest.fingerprint, archive=newer_archive, manifest=newer
    )
    # An external editor wrote a THIRD tree into the alias during the crash gap.
    third = _source(tmp_path / "third", "# Third\n")
    os.replace(third, storage.path(manifest.name))
    third_hash = tree_hash_of(storage.path(manifest.name))

    storage.recover()

    assert tree_hash_of(storage.path(manifest.name)) == third_hash
    record = storage.journal.read(operation)
    assert record.phase == "blocked" and record.error_class == "local-concurrent-edit"


def test_document_recovery_ignores_bundle_transactions_and_the_reader_recovers_both(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        _baseline(identity, manifest),
        expected_fingerprint=None,
    )
    newer_archive, newer = _bundle(tmp_path / "next", "# New body\n")
    operation = _crash_between_renames(
        storage, identity, before=manifest.fingerprint, archive=newer_archive, manifest=newer
    )

    # The document adapter must not settle a transaction it cannot interpret.
    ManagedSkillStorage(root).recover()
    assert storage.journal.read(operation).phase == "prepared"

    with managed_skill_read(root):
        pass

    assert storage.journal.read(operation).phase == "aborted"
    assert tree_hash_of(storage.path(manifest.name)) == manifest.fingerprint


def test_a_recorder_transaction_of_the_other_adapter_is_not_touched(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    storage = ManagedBundleStorage(root)
    document = SkillDocument.from_json(
        {
            "document_version": 1,
            "name": "pdf-forms",
            "description": "Fill and validate PDF forms.",
            "files": [
                {
                    "path": "SKILL.md",
                    "content": (
                        "---\nname: pdf-forms\ndescription: Fill and validate PDF forms." "\n---\n"
                    ),
                }
            ],
        }
    )
    transaction = storage.journal.prepare(
        identity,
        "pull",
        before_fingerprint=None,
        after_fingerprint=document.fingerprint,
        payload={"adapter": "managed-skill", "local_alias": document.name},
    )

    storage.recover()

    assert storage.journal.read(transaction.operation_id).phase == "prepared"


def test_adapter_snapshot_reads_a_bundle_link_by_tree_hash(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    identity = _identity()
    archive, manifest = _bundle(tmp_path)
    storage = ManagedBundleStorage(root)
    storage.apply(
        identity,
        manifest.name,
        archive,
        manifest,
        _baseline(identity, manifest),
        expected_fingerprint=None,
    )

    snapshot = SkillAdapter().snapshot(root, identity.local_id)
    assert snapshot.name == manifest.name
    assert snapshot.fields == manifest.to_json()
    assert snapshot.fingerprint == manifest.bundle.tree_hash

    (storage.path(manifest.name) / "references" / "guide.md").write_text(
        "edited\n", encoding="utf-8"
    )
    changed = SkillAdapter().snapshot(root, identity.local_id)
    assert changed.fingerprint != manifest.bundle.tree_hash


def test_fingerprint_dispatch_keeps_document_semantics(tmp_path: Path) -> None:
    _archive, manifest = _bundle(tmp_path)
    assert prov.fingerprint_fields("skill", manifest.to_json()) == manifest.fingerprint
    assert prov.fingerprint_snapshot("skill", manifest.to_json()) == manifest.bundle.tree_hash

    document = SkillDocument.from_json(
        {
            "document_version": 1,
            "name": "pdf-forms",
            "description": "Fill and validate PDF forms.",
            "files": [
                {
                    "path": "SKILL.md",
                    "content": (
                        "---\nname: pdf-forms\ndescription: Fill and validate PDF forms." "\n---\n"
                    ),
                }
            ],
        }
    )
    assert prov.fingerprint_fields("skill", document.to_json()) == document.fingerprint


def test_the_union_helpers_dispatch_on_the_manifest_form(tmp_path: Path) -> None:
    _archive, manifest = _bundle(tmp_path)
    parsed = skill_manifest_from_json(manifest.to_json())
    assert isinstance(parsed, SkillBundleManifest) and parsed == manifest
    envelope = dict(manifest.to_json())
    # A hub DETAIL payload carries metadata beside the manifest; the tolerant
    # constructor takes the manifest subset and still validates it strictly.
    envelope.update({"id": "sk-1", "revision": '"rev"', "sync_contract": {"skill_bundles": 1}})
    assert skill_manifest_from_remote(envelope) == manifest
    document = SkillDocument.from_json(
        {
            "document_version": 1,
            "name": "pdf-forms",
            "description": "Fill and validate PDF forms.",
            "files": [
                {
                    "path": "SKILL.md",
                    "content": (
                        "---\nname: pdf-forms\ndescription: Fill and validate PDF forms." "\n---\n"
                    ),
                }
            ],
        }
    )
    assert skill_manifest_from_json(document.to_json()) == document


def test_the_adapter_validates_both_forms_and_refuses_a_mixed_one(tmp_path: Path) -> None:
    adapter = SkillAdapter()
    _archive, manifest = _bundle(tmp_path)
    adapter.validate(manifest.to_json())
    fields = manifest.to_json()
    fields["files"] = []
    with pytest.raises(ValueError):
        adapter.validate(fields)
    with pytest.raises(ValueError):
        adapter.canonical_fields({"bundle": {"format": "zip"}})
