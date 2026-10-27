"""``ManagedSkillStorage.author``: a LOCAL edit that must never move the link baseline.

The invariant under test is the reason the method exists: an edit of a hub-linked
skill leaves its baseline (and policy) exactly where they were, because the gap
between baseline and folder is what the link check reports as ``local-only`` and
the outbound path merges. Every cell runs against real directories and the real
journal; the only seam is the checkpoint hook the existing crash tests use.
"""

import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import skill_storage
from local_operator.hub_sync.identity import LinkError, LinkIdentity
from local_operator.hub_sync.provenance import make_record, read_baseline
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage
from local_operator.hub_sync.transactions import TransactionError

from .test_skill_document import manifest


def _doc(body: str = "# Body\n", *, script: str | None = None) -> SkillDocument:
    files = [{"path": "scripts/doctor.sh", "content": script or "#!/bin/sh\necho ok\n"}]
    value = manifest(files=files)
    value["files"][0]["content"] = value["files"][0]["content"].replace("# Body\n", body)
    return SkillDocument.from_json(value)


def _linked(tmp_path: Path):
    """A pulled, linked skill: folder + baseline written the way ``apply`` writes them."""
    identity = LinkIdentity(
        "skill", "immutable-local", "https://example.invalid/v1", "org", "tenant-1", "remote"
    )
    document = _doc()
    baseline = replace(
        make_record(
            "skill", identity.local_id, identity.hub_id, "tenant-1", document.to_json(), "pull"
        ),
        identity=identity,
        remote_revision='"revision-1"',
        lineage="clean",
        local_alias=document.name,
    )
    storage = ManagedSkillStorage(tmp_path / "profile")
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)
    return storage, identity, document, baseline


def test_create_then_update_an_unlinked_skill(tmp_path: Path) -> None:
    storage = ManagedSkillStorage(tmp_path / "profile")
    first = _doc()
    created = storage.author(first.name, first, None)
    assert created.changed and not created.linked and created.baseline_fingerprint is None
    assert SkillDocument.from_directory(storage.path(first.name)) == first

    second = _doc("# Edited\n")
    updated = storage.author(second.name, second, first.fingerprint)
    assert updated.before_fingerprint == first.fingerprint
    assert SkillDocument.from_directory(storage.path(first.name)) == second
    # No baseline was invented for a purely local skill.
    assert prov.alias_owner(storage.config_dir, "skill", first.name) is None


def test_editing_a_linked_skill_leaves_baseline_and_policy_untouched(tmp_path: Path) -> None:
    storage, identity, original, _baseline = _linked(tmp_path)
    before_record = read_baseline(storage.config_dir, "skill", identity.local_id)
    assert before_record is not None
    baseline_file = prov.baseline_path(storage.config_dir, "skill", identity.local_id)
    config_file = storage.config_dir / "config.yml"
    baseline_bytes = baseline_file.read_bytes()
    config_bytes = config_file.read_bytes() if config_file.exists() else None

    edited = _doc("# Edited locally\n")
    result = storage.author(edited.name, edited, original.fingerprint)

    assert result.linked and result.changed
    assert result.baseline_fingerprint == before_record.fingerprint == original.fingerprint
    assert result.fingerprint == edited.fingerprint != original.fingerprint
    # Byte-identical: not "equal after parsing" -- the record was not rewritten.
    assert baseline_file.read_bytes() == baseline_bytes
    assert (config_file.read_bytes() if config_file.exists() else None) == config_bytes
    assert read_baseline(storage.config_dir, "skill", identity.local_id) == before_record


def test_stale_fingerprint_is_refused_and_the_bytes_are_untouched(tmp_path: Path) -> None:
    storage, _identity, original, _baseline = _linked(tmp_path)
    live = storage.path(original.name)
    snapshot = {p: p.read_bytes() for p in live.rglob("*") if p.is_file()}

    with pytest.raises(LinkError) as caught:
        storage.author(original.name, _doc("# Stale write\n"), "0" * 64)

    assert caught.value.code == "local-concurrent-edit"
    assert {p: p.read_bytes() for p in live.rglob("*") if p.is_file()} == snapshot
    assert storage.journal.pending(kind="skill") == []


def test_create_over_an_existing_name_or_a_link_is_alias_occupied(tmp_path: Path) -> None:
    storage, _identity, original, _baseline = _linked(tmp_path)
    with pytest.raises(LinkError) as caught:
        storage.author(original.name, _doc("# Other\n"), None)
    assert caught.value.code == "alias-occupied"

    plain = ManagedSkillStorage(tmp_path / "other")
    plain.author(original.name, original, None)
    with pytest.raises(LinkError) as again:
        plain.author(original.name, _doc("# Other\n"), None)
    assert again.value.code == "alias-occupied"


def test_replaying_an_applied_update_is_a_noop_success(tmp_path: Path) -> None:
    storage = ManagedSkillStorage(tmp_path / "profile")
    first, second = _doc(), _doc("# Two\n")
    storage.author(first.name, first, None)
    storage.author(second.name, second, first.fingerprint)
    # A retry after a lost response: the folder already holds the requested bytes.
    replay = storage.author(second.name, second, first.fingerprint)
    assert replay.changed is False and replay.fingerprint == second.fingerprint


def test_an_unpulled_or_bundle_link_is_not_editable(tmp_path: Path) -> None:
    storage, identity, original, _baseline = _linked(tmp_path)
    storage.unpull(identity.local_id)
    with pytest.raises(LinkError) as caught:
        storage.author(original.name, _doc("# Edit\n"), original.fingerprint)
    assert caught.value.code == "skill-not-editable"


def test_a_symlinked_root_or_alias_is_refused_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    config = tmp_path / "profile"
    config.mkdir()
    (config / "skills").symlink_to(outside)
    storage = ManagedSkillStorage(config)
    with pytest.raises(LinkError):
        storage.author(_doc().name, _doc(), None)
    assert list(outside.iterdir()) == []

    (config / "skills").unlink()
    (config / "skills").mkdir()
    (config / "skills" / "example-skill").symlink_to(outside)
    with pytest.raises(LinkError):
        storage.author("example-skill", _doc(), None)
    assert list(outside.iterdir()) == []


def test_a_symlink_inside_the_current_skill_is_refused(tmp_path: Path) -> None:
    storage = ManagedSkillStorage(tmp_path / "profile")
    document = _doc()
    storage.author(document.name, document, None)
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    (storage.path(document.name) / "references").mkdir()
    (storage.path(document.name) / "references" / "leak.md").symlink_to(secret)
    with pytest.raises(Exception):  # noqa: B017 - from_directory names the symlink refusal
        storage.author(document.name, _doc("# Edit\n"), document.fingerprint)
    assert secret.read_text() == "private"


def test_an_executable_bit_survives_an_edit(tmp_path: Path) -> None:
    """The #2868 rule: a UI/model edit must not silently strip ``scripts/doctor.sh``'s +x."""
    storage = ManagedSkillStorage(tmp_path / "profile")
    document = _doc()
    storage.author(document.name, document, None)
    script = storage.path(document.name) / "scripts" / "doctor.sh"
    # Fresh authored files land inert (0644); the user then makes it executable.
    assert stat.S_IMODE(script.stat().st_mode) == 0o644
    os.chmod(script, 0o755)
    current = SkillDocument.from_directory(storage.path(document.name))

    edited = _doc("# Edited\n", script="#!/bin/sh\necho changed\n")
    storage.author(edited.name, edited, current.fingerprint)

    assert script.read_text() == "#!/bin/sh\necho changed\n"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    # Group/other WRITE never rides along (the same sanitiser as a pull).
    os.chmod(script, 0o777)
    again = SkillDocument.from_directory(storage.path(document.name))
    storage.author(edited.name, _doc("# Third\n", script="#!/bin/sh\necho 3\n"), again.fingerprint)
    assert stat.S_IMODE(script.stat().st_mode) == 0o755


@pytest.mark.parametrize("crash_at", ["staged", "old-renamed", "new-renamed", "content-committed"])
def test_a_crash_at_every_phase_recovers_to_a_whole_skill(tmp_path: Path, crash_at: str) -> None:
    """The crash tests' discipline, applied to the new basis, including the two-rename gap."""
    storage, identity, original, _baseline = _linked(tmp_path)
    before_record = read_baseline(storage.config_dir, "skill", identity.local_id)
    edited = _doc("# Edited, then the process died\n")

    def crash(phase: str) -> None:
        if phase == crash_at:
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        storage.author(edited.name, edited, original.fingerprint, checkpoint=crash)

    # What the next reader does: recovery runs before any read.
    storage.recover()

    landed = storage._fingerprint_path(storage.path(edited.name))
    # "old-renamed" is the gap: the alias is absent until recovery puts the
    # VERIFIED original back. After "new-renamed" the edit is whole on disk and is
    # kept (finalize is a no-op for this basis, so nothing else needs to run).
    expected = original.fingerprint if crash_at in ("staged", "old-renamed") else edited.fingerprint
    assert landed == expected
    assert storage.journal.pending(kind="skill", local_id=identity.local_id) == []
    # Whatever recovery decided, the baseline was never moved.
    assert read_baseline(storage.config_dir, "skill", identity.local_id) == before_record


def test_a_live_transaction_blocks_a_second_author(tmp_path: Path) -> None:
    storage, _identity, original, _baseline = _linked(tmp_path)
    edited = _doc("# first\n")
    seen: list[str] = []

    def hold(phase: str) -> None:
        if phase == "staged":
            second = _doc("# second\n")
            # The first author is mid-flight and holds its lease: the second must
            # refuse rather than swap a directory the first is staging for.
            with pytest.raises(TransactionError):
                storage.author(second.name, second, original.fingerprint)
            seen.append("refused")

    storage.author(edited.name, edited, original.fingerprint, checkpoint=hold)
    assert seen == ["refused"]
    assert SkillDocument.from_directory(storage.path(edited.name)) == edited


def test_remove_local_moves_to_a_kept_backup(tmp_path: Path) -> None:
    storage = ManagedSkillStorage(tmp_path / "profile")
    document = _doc()
    storage.author(document.name, document, None)
    backup = storage.remove_local(document.name)
    assert not storage.path(document.name).exists()
    assert backup.parent == storage.config_dir / "hub" / "backups"
    assert backup.name.startswith(f"skill-local-{document.name}-")
    assert SkillDocument.from_directory(backup) == document


def test_remove_local_refuses_a_linked_alias(tmp_path: Path) -> None:
    storage, _identity, original, _ = _linked(tmp_path)
    with pytest.raises(LinkError) as caught:
        storage.remove_local(original.name)
    assert caught.value.code == "origin-conflict"
    assert storage.path(original.name).is_dir()


def test_a_forged_local_edit_basis_is_not_trusted(tmp_path: Path) -> None:
    """m5: ``basis`` skips baseline settlement, so only an ``author``-shaped record may claim it."""
    storage, identity, original, baseline = _linked(tmp_path)
    forged = storage.journal.prepare(
        identity,
        "pull",
        before_fingerprint=original.fingerprint,
        after_fingerprint=original.fingerprint,
        payload={
            "adapter": "managed-skill",
            "local_alias": original.name,
            "baseline": baseline.to_json(),
        },
        basis=skill_storage.LOCAL_EDIT_BASIS,
    )
    with pytest.raises(TransactionError, match="malformed"):
        storage._finalize(forged)
