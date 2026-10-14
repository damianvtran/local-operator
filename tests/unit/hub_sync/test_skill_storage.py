"""Narrow journal/owned-reader guards; not engine or remote publication coverage."""

from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.config_writer import nonblocking_file_claim
from local_operator.hub_sync import skill_storage
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.hub_sync.provenance import make_record, read_baseline
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import (
    ManagedSkillStorage,
    managed_skill_read,
)
from local_operator.hub_sync.transactions import LocalTransaction, TransactionError
from local_operator.skills.discovery import (
    discover_skills,
    managed_root_context,
    roots_fingerprint,
)
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


def test_fingerprint_cannot_recover_external_symlinked_profile(tmp_path: Path) -> None:
    """R3: the fingerprint path must apply the SAME world containment as discovery.

    A bounded session reaches its native root only through the world. A world-
    external profile addressed by an in-world symlink must not be locked,
    recovered, or have a prepared journal completed merely because the
    fingerprint stat'd it first.
    """
    world = tmp_path / "world"
    world.mkdir()
    external = tmp_path / "external-profile"
    skill_dir = external / "skills" / "example-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: example-skill\ndescription: Example\n---\n# B\n"
    )
    link = world / "skills"
    link.symlink_to(external / "skills")

    assert roots_fingerprint([link], config_dir=external, boundary=world) is not None
    assert not external.exists() or not (external / "hub").exists()


def test_roots_fingerprint_still_locks_an_in_world_native_root(tmp_path: Path) -> None:
    """R3 positive control: an IN-world native root still uses read discipline."""
    world = tmp_path / "world"
    native = world / "profile" / "skills"
    (native / "example-skill").mkdir(parents=True)
    (native / "example-skill" / "SKILL.md").write_text(
        "---\nname: example-skill\ndescription: Example\n---\n# B\n"
    )
    assert roots_fingerprint([native], config_dir=world / "profile", boundary=world) is not None


def test_differing_origin_cannot_claim_a_temporarily_absent_alias(tmp_path: Path) -> None:
    """R5: a crash gap must not let another ORG origin take the PUBLIC owner's alias."""
    public_identity, public_doc, public_baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(
        public_identity, public_doc.name, public_doc, public_baseline, expected_fingerprint=None
    )

    def crash(phase: str) -> None:
        if phase == "old-renamed":
            raise RuntimeError("synthetic crash")

    _, changed, next_baseline = _input("# New body\n")
    with pytest.raises(RuntimeError, match="synthetic crash"):
        storage.apply(
            public_identity,
            changed.name,
            changed,
            next_baseline,
            expected_fingerprint=public_doc.fingerprint,
            checkpoint=crash,
        )
    destination, _, _ = storage._locations(storage.journal.pending(kind="skill")[0])
    assert not destination.exists()

    org_identity = LinkIdentity(
        "skill", "org-local", "https://example.invalid/v1", "org", "org-tab", "remote"
    )
    # A well-formed ORG baseline that would otherwise validate, but claims the
    # PUBLIC owner's exact alias while that owner sits in the crash gap.
    org_baseline = replace(
        public_baseline,
        kind="skill",
        local_id="org-local",
        hub_id="remote",
        tenant_id="org-tab",
        identity=org_identity,
        local_alias=public_doc.name,
    )
    with pytest.raises(Exception) as excinfo:
        storage.apply(
            org_identity, public_doc.name, public_doc, org_baseline, expected_fingerprint=None
        )
    assert "alias belongs to another" in str(excinfo.value)


def test_rename_ancestors_are_fsynced_before_swap(tmp_path: Path, monkeypatch) -> None:
    """R4: the backup/staging parent directory entries are fsynced exactly at swap."""
    calls: list[Path] = []
    real = skill_storage._sync_ancestors

    def spy(directory: Path, stop: Path) -> None:
        calls.append(Path(directory))
        real(directory, stop)

    monkeypatch.setattr(skill_storage, "_sync_ancestors", spy)
    identity, document, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)
    hub = storage.config_dir / "hub"
    # The exact parents whose directory entries change at the swap: the backup
    # collection and the staged operation's parent.
    assert (hub / "backups") in calls
    assert any(path.parent == hub / "staging" for path in calls)


def test_check_hub_skills_reports_available_when_the_remote_manifest_moved(tmp_path: Path) -> None:
    """The skill check path: a linked skill compared against a changed remote manifest."""
    from local_operator.hub_sync import check as chk

    identity, document, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)

    # A remote manifest whose body differs from the linked baseline.
    changed = manifest("# REMOTE move\n")

    class _Client:
        def get_skill(self, skill_id, **_kw):
            # The server returns the strict manifest; the adapter canonicalizes it.
            return changed

    checks = chk.check_hub_skills(
        config_dir=storage.config_dir,
        client_for_tenant=lambda _t: _Client(),
    )
    assert len(checks) == 1
    check = checks[0]
    assert check.kind == "skill" and check.local_id == identity.local_id
    assert check.verdict == "available"
    assert check.remote is not None and "REMOTE move" in check.remote["files"][0]["content"]


def test_check_hub_skills_is_unavailable_without_a_client(tmp_path: Path) -> None:
    from local_operator.hub_sync import check as chk

    identity, document, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)

    checks = chk.check_hub_skills(config_dir=storage.config_dir, client_for_tenant=lambda _t: None)
    assert checks and checks[0].reason == "no-credential"
