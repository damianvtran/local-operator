"""Raw adoption CAS, durability and crash recovery through the storage adapter."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.provenance import read_baseline
from local_operator.hub_sync.skill_adoption import (
    AdoptionError,
    build_proposal,
    durable_snapshot,
    read_tree,
)
from local_operator.hub_sync.skill_storage import ManagedSkillStorage

from .test_skill_storage import _input


def _setup(tmp_path):
    identity, document, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "profile")
    folder = storage.path(document.name)
    for file in document.files:
        path = folder / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file.content)
    (folder / "assets").mkdir()
    (folder / "assets/local.bin").write_bytes(b"\x00\xfforiginal")
    tree = read_tree(folder)
    snapshot = durable_snapshot(storage.config_dir, document.name, tree)
    return identity, document, baseline, storage, tree, snapshot


def test_keep_raw_no_content_write_and_binary_survives_update(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    inode = tree.root.stat().st_ino
    transaction = storage.adopt(
        identity,
        document.name,
        None,
        baseline,
        expected_tree_hash=tree.tree_hash,
        snapshot=snapshot,
        carried_paths=("assets/local.bin",),
    )
    assert transaction.phase == "complete"
    assert read_tree(tree.root).tree_hash == tree.tree_hash
    assert tree.root.stat().st_ino == inode
    assert storage.snapshot(identity.local_id).document == document
    changed = document.with_description("Updated description")
    changed_baseline = replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint)
    storage.apply(
        identity,
        document.name,
        changed,
        changed_baseline,
        expected_fingerprint=document.fingerprint,
    )
    assert (tree.root / "assets/local.bin").read_bytes() == b"\x00\xfforiginal"
    assert storage.snapshot(identity.local_id).document == changed


def test_raw_tree_cas_includes_asset_and_chmod(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    (tree.root / "assets/local.bin").chmod(0o600)
    with pytest.raises(LinkError) as error:
        storage.adopt(
            identity,
            document.name,
            document,
            baseline,
            expected_tree_hash=tree.tree_hash,
            snapshot=snapshot,
        )
    assert error.value.code == "local-concurrent-edit"
    assert read_baseline(storage.config_dir, "skill", identity.local_id) is None
    assert storage._projection(document.name) is None


@pytest.mark.parametrize("phase", ["prepared", "old-renamed", "new-renamed", "content-committed"])
def test_crash_recovery_preserves_original_or_complete_result(tmp_path, phase):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    changed = document.with_description("Updated description")

    def crash(current):
        if current == phase:
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        storage.adopt(
            identity,
            document.name,
            changed,
            baseline,
            expected_tree_hash=tree.tree_hash,
            snapshot=snapshot,
            carried_paths=("assets/local.bin",),
            checkpoint=crash,
        )
    storage.recover()
    assert (tree.root / "assets/local.bin").read_bytes() == b"\x00\xfforiginal"
    assert read_tree(Path(snapshot["path"]) / "content").tree_hash == tree.tree_hash
    if phase in ("prepared", "old-renamed"):
        assert read_tree(tree.root).tree_hash == tree.tree_hash
        assert read_baseline(storage.config_dir, "skill", identity.local_id) is None
    else:
        assert storage.snapshot(identity.local_id).document == changed


def test_raw_reader_rejects_symlink(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "outside").symlink_to(tmp_path)
    with pytest.raises(AdoptionError):
        read_tree(folder)


def test_no_model_description_and_metadata_never_take_hub_silently(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    skill = tree.root / "SKILL.md"
    skill.write_text(
        skill.read_text().replace(
            "description:",
            "metadata:\n  local-rule: Keep every audit record\ndescription:",
        )
    )
    proposal = build_proposal(
        document.name, read_tree(tree.root), document, resolver=None, allow_llm=False
    )
    assert proposal.document is not None
    assert "Keep every audit record" in proposal.document.to_json()["files"][0]["content"]
    local = skill.read_text().replace(document.description, "Unique local routing guidance")
    skill.write_text(local)
    proposal = build_proposal(
        document.name, read_tree(tree.root), document, resolver=None, allow_llm=False
    )
    assert proposal.state == "needs-choice"
    assert proposal.document is None


def test_full_diffs_and_keep_local_preview_are_truthful(tmp_path):
    from local_operator.hub_sync.skill_adoption import _diff
    from local_operator.hub_sync.skill_document import SkillFile

    late = "Late unique routing obligation"
    diff = _diff("old\n" * 20000 + late, "new\n" * 20000, "local", "proposed", "large.md")
    assert late in diff
    assert len(diff) > 40000
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    remote = replace(document, files=document.files + (SkillFile("references/hub-only.md", "Hub"),))
    proposal = build_proposal(document.name, tree, remote, resolver=None, strategy="keep-local")
    hub_only = next(plan for plan in proposal.files if plan.path == "references/hub-only.md")
    assert hub_only.action == "not-installed"
    assert hub_only.proposed is None
    assert "-Hub" in hub_only.to_json()["diff_vs_hub"]
    assert read_tree(tree.root).tree_hash == tree.tree_hash


def test_relocation_collision_never_drops_unique_root_guidance(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    (tree.root / "requirements.md").write_text("Unique original root obligation")
    (tree.root / "references").mkdir()
    (tree.root / "references/requirements.md").write_text("Distinct reference obligation")
    proposal = build_proposal(
        document.name, read_tree(tree.root), document, resolver=None, allow_llm=False
    )
    assert proposal.state == "needs-choice"


def test_text_assets_never_reach_resolver(tmp_path):
    from local_operator.hub_sync.skill_document import SkillFile

    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    (tree.root / "assets/payload.txt").write_text("Unique local payload")
    remote = replace(
        document,
        files=document.files + (SkillFile("assets/payload.txt", "Hub payload"),),
    )

    class RejectCalls:
        def resolve(self, req):
            raise AssertionError("asset dispatched to model")

    proposal = build_proposal(document.name, read_tree(tree.root), remote, resolver=RejectCalls())
    assert proposal.state == "needs-choice"
    assert any(plan.path == "assets/payload.txt" for plan in proposal.unresolved)


@pytest.mark.parametrize("mutate_after_preview", [False, True])
def test_subscribe_then_adopt_existing_authored_folder(tmp_path, mutate_after_preview):
    from local_operator.hub_sync import service

    from .test_skill_state import ALIAS, HUB_ID, _rig, _subscribe

    ctx, manager, remote, base, payload = _rig(tmp_path)
    subscribed = _subscribe(ctx, base)
    folder = manager.config_dir / "skills" / ALIAS
    for file in payload["files"]:
        path = folder / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file["content"])
    before = read_tree(folder).tree_hash
    arguments: dict[str, Any] = dict(
        hub_source=base,
        hub_id=HUB_ID,
        visibility="public",
        tenant_id=None,
        alias=ALIAS,
    )
    preview = service.install_skill(ctx, dry_run=True, **arguments)
    storage = ManagedSkillStorage(manager.config_dir)
    assert storage._projection(ALIAS) is None  # cancellation does not authorize exclusions
    if mutate_after_preview:
        main = folder / "SKILL.md"
        main.write_text(main.read_text() + "\nLater user guidance.\n")
        with pytest.raises(LinkError):
            service.install_skill(
                ctx, confirm=True, adoption_receipt=preview["adoption_receipt"], **arguments
            )
        assert storage._projection(ALIAS) is None
        record = read_baseline(manager.config_dir, "skill", subscribed["link"]["local_id"])
        assert record is not None and record.content_state == "absent"
        return
    result = service.install_skill(
        ctx,
        confirm=True,
        adoption_receipt=preview["adoption_receipt"],
        **arguments,
    )
    assert result["reports"][0]["applied"]
    assert read_tree(folder).tree_hash == before
    record = read_baseline(manager.config_dir, "skill", subscribed["link"]["local_id"])
    assert record is not None and record.content_state == "present"


def _adopted(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    storage.adopt(
        identity,
        document.name,
        None,
        baseline,
        expected_tree_hash=tree.tree_hash,
        snapshot=snapshot,
        carried_paths=("assets/local.bin",),
    )
    return identity, document, baseline, storage, tree, snapshot


@pytest.mark.parametrize("writer", ["apply", "author"])
@pytest.mark.parametrize("phase", ["staged", "old-renamed", "new-renamed"])
@pytest.mark.parametrize("edit", ["binary", "mode"])
def test_ordinary_writers_refuse_raw_edits_without_settlement(tmp_path, writer, phase, edit):
    identity, document, baseline, storage, tree, snapshot = _adopted(tmp_path)
    changed = document.with_description("Updated remote routing")
    new_baseline = replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint)
    prior_baseline = read_baseline(storage.config_dir, "skill", identity.local_id)
    config = (storage.config_dir / "config.yml").read_bytes()
    edited_path = None

    def interfere(current):
        nonlocal edited_path
        if current != phase:
            return
        transaction = storage.journal.pending(kind="skill")[0]
        destination, staged, backup = storage._locations(transaction)
        folder = backup if phase == "old-renamed" else destination
        edited_path = folder / "assets/local.bin"
        if edit == "binary":
            edited_path.write_bytes(b"\x00LATER EDIT")
        else:
            edited_path.chmod(0o600)

    with pytest.raises(LinkError) as failure:
        if writer == "apply":
            storage.apply(
                identity,
                document.name,
                changed,
                new_baseline,
                expected_fingerprint=document.fingerprint,
                checkpoint=interfere,
            )
        else:
            storage.author(
                document.name,
                changed,
                expected_fingerprint=document.fingerprint,
                checkpoint=interfere,
            )
    assert failure.value.code == "local-concurrent-edit"
    storage.recover()
    assert read_baseline(storage.config_dir, "skill", identity.local_id) == prior_baseline
    assert (storage.config_dir / "config.yml").read_bytes() == config
    # An old-renamed refusal restores the edited original, not its stale snapshot.
    actual = tree.root / "assets/local.bin"
    if edit == "binary":
        assert actual.read_bytes() == b"\x00LATER EDIT"
    else:
        assert actual.stat().st_mode & 0o777 == 0o600
    assert read_tree(Path(snapshot["path"]) / "content").tree_hash == tree.tree_hash


@pytest.mark.parametrize("writer", ["apply", "author"])
def test_recovery_does_not_finalize_after_binary_edit_to_new_tree(tmp_path, writer):
    identity, document, baseline, storage, tree, snapshot = _adopted(tmp_path)
    changed = document.with_description("Updated remote routing")
    new_baseline = replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint)
    prior = read_baseline(storage.config_dir, "skill", identity.local_id)

    def crash(phase):
        if phase == "new-renamed":
            (tree.root / "assets/local.bin").write_bytes(b"\x00after crash edit")
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        if writer == "apply":
            storage.apply(
                identity,
                document.name,
                changed,
                new_baseline,
                expected_fingerprint=document.fingerprint,
                checkpoint=crash,
            )
        else:
            storage.author(
                document.name, changed, expected_fingerprint=document.fingerprint, checkpoint=crash
            )
    storage.recover()
    assert read_baseline(storage.config_dir, "skill", identity.local_id) == prior
    assert (tree.root / "assets/local.bin").read_bytes() == b"\x00after crash edit"
    assert storage.journal.pending(kind="skill")[0].phase == "blocked"


@pytest.mark.parametrize("path", ["scripts/inert.py", "assets/payload.txt", "references/data.txt"])
def test_script_multiline_literal_requires_choice_despite_hub_only_file(tmp_path, path):
    from local_operator.hub_sync.skill_document import SkillFile

    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    local = 'VALUE = """first line\nsecond line"""\n'
    remote = 'VALUE = """first line second line"""\n'
    (tree.root / path).parent.mkdir(parents=True, exist_ok=True)
    (tree.root / path).write_text(local)
    files = tuple(file for file in document.files if file.path != path) + (
        SkillFile(path, remote),
        SkillFile("references/hub-only.md", "Hub addition"),
    )
    hub = replace(document, files=files)

    class NeverResolve:
        def resolve(self, req):
            raise AssertionError("data dispatched to resolver")

    proposal = build_proposal(document.name, read_tree(tree.root), hub, resolver=NeverResolve())
    assert proposal.state == "needs-choice"
    assert proposal.document is None
    assert (tree.root / path).read_text() == local
    assert any(plan.path == "references/hub-only.md" for plan in proposal.files)


def test_source_relative_relocation_links_and_escape_refusal(tmp_path):
    from local_operator.hub_sync.skill_document import SkillFile

    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    (tree.root / "requirements.md").write_text("# Requirements\n")
    (tree.root / "references").mkdir()
    (tree.root / "references/local.md").write_text(
        "Read [root](../requirements.md) and [same](./local.md).\n"
    )
    (tree.root / "rubric.md").write_text("Read [requirements](./requirements.md).\n")
    remote = replace(
        document,
        files=document.files
        + (
            SkillFile("references/requirements.md", "# Requirements\n"),
            SkillFile("references/rubric.md", "Read [requirements](./requirements.md).\n"),
        ),
    )
    proposal = build_proposal(
        document.name, read_tree(tree.root), remote, resolver=None, allow_llm=False
    )
    assert proposal.document is not None
    files = {file.path: file.content for file in proposal.document.files}
    assert "[root](requirements.md)" in files["references/local.md"]
    assert "[same](./local.md)" in files["references/local.md"]
    assert "[requirements](./requirements.md)" in files["references/rubric.md"]
    for invalid in ("../../requirements.md", "requirements.md"):
        (tree.root / "references/local.md").write_text(f"Read {invalid}.\n")
        with pytest.raises(AdoptionError) as failure:
            build_proposal(document.name, read_tree(tree.root), remote, resolver=None)
        assert failure.value.code == "adoption-needs-choice"


@pytest.mark.parametrize("crash", [False, True])
def test_unpull_retires_exclusions_and_fresh_pull_accepts_former_asset(
    tmp_path,
    monkeypatch,
    crash,
):
    from local_operator.hub_sync.skill_document import SkillFile

    identity, document, baseline, storage, tree, snapshot = _adopted(tmp_path)
    retire = storage._retire_projection
    if crash:

        def stop(alias):
            raise RuntimeError("crash")

        monkeypatch.setattr(storage, "_retire_projection", stop)
        with pytest.raises(RuntimeError, match="crash"):
            storage.unpull(identity.local_id)
        monkeypatch.setattr(storage, "_retire_projection", retire)
        storage.recover_unpull(local_id=identity.local_id)
    else:
        storage.unpull(identity.local_id)
    assert storage._projection(document.name) is None
    hub = replace(
        document,
        files=document.files + (SkillFile("assets/local.bin", "New Hub asset"),),
    )
    record = replace(baseline, fields=hub.to_json(), fingerprint=hub.fingerprint)
    storage.apply(identity, document.name, hub, record, expected_fingerprint=None)
    assert storage.snapshot(identity.local_id).document == hub
    assert (tree.root / "assets/local.bin").read_text() == "New Hub asset"


def test_unpull_then_authored_folder_can_be_adopted_again(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _adopted(tmp_path)
    storage.unpull(identity.local_id)
    tree.root.mkdir()
    for file in document.files:
        path = tree.root / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file.content)
    main = tree.root / "SKILL.md"
    main.write_text(main.read_text() + "\nUnique post-unpull authored guidance.\n")
    expected = storage._document(document.name)
    raw = read_tree(tree.root)
    backup = durable_snapshot(storage.config_dir, document.name, raw)
    assert storage._projection(document.name) is None  # cancelled preview has no side effect
    storage.adopt(
        identity, document.name, None, baseline, expected_tree_hash=raw.tree_hash, snapshot=backup
    )
    assert storage.snapshot(identity.local_id).document == expected
    assert read_tree(tree.root).tree_hash == raw.tree_hash


@pytest.mark.parametrize("writer", ["apply", "author"])
@pytest.mark.parametrize("edit", ["binary", "mode"])
def test_staged_asset_tamper_is_refused(tmp_path, writer, edit):
    identity, document, baseline, storage, tree, snapshot = _adopted(tmp_path)
    changed = document.with_description("Updated remote routing")
    prior = read_baseline(storage.config_dir, "skill", identity.local_id)

    def tamper(phase):
        if phase == "staged":
            transaction = storage.journal.pending(kind="skill")[0]
            asset = storage._locations(transaction)[1] / "assets/local.bin"
            if edit == "binary":
                asset.write_bytes(b"\x00tampered")
            else:
                asset.chmod(0o600)

    with pytest.raises(LinkError) as failure:
        if writer == "apply":
            record = replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint)
            storage.apply(
                identity,
                document.name,
                changed,
                record,
                expected_fingerprint=document.fingerprint,
                checkpoint=tamper,
            )
        else:
            storage.author(
                document.name, changed, expected_fingerprint=document.fingerprint, checkpoint=tamper
            )
    assert failure.value.code == "local-concurrent-edit"
    assert read_tree(tree.root).tree_hash == tree.tree_hash
    assert read_baseline(storage.config_dir, "skill", identity.local_id) == prior


@pytest.mark.parametrize("phase", ["old-renamed", "new-renamed"])
@pytest.mark.parametrize("edit", ["binary", "mode"])
def test_crash_recovery_preserves_unexpected_raw_edits(tmp_path, phase, edit):
    identity, document, baseline, storage, tree, snapshot = _adopted(tmp_path)
    changed = document.with_description("Updated remote routing")
    prior = read_baseline(storage.config_dir, "skill", identity.local_id)

    def crash(current):
        if current == phase:
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        storage.apply(
            identity,
            document.name,
            changed,
            replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint),
            expected_fingerprint=document.fingerprint,
            checkpoint=crash,
        )
    transaction = storage.journal.pending(kind="skill")[0]
    destination, staged, backup = storage._locations(transaction)
    asset = (backup if phase == "old-renamed" else destination) / "assets/local.bin"
    if edit == "binary":
        asset.write_bytes(b"\x00post-crash edit")
    else:
        asset.chmod(0o600)
    edited = read_tree(asset.parent.parent).tree_hash
    storage.recover()
    assert read_tree(asset.parent.parent).tree_hash == edited
    assert read_baseline(storage.config_dir, "skill", identity.local_id) == prior
    settled = storage.journal.read(transaction.operation_id)
    assert settled is not None and settled.phase == "blocked"
