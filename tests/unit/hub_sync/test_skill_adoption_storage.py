"""Raw adoption CAS, durability and crash recovery through the storage adapter."""
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.provenance import read_baseline
from local_operator.hub_sync.skill_adoption import (
    AdoptionError, build_proposal, durable_snapshot, read_tree,
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
        identity, document.name, None, baseline,
        expected_tree_hash=tree.tree_hash, snapshot=snapshot, carried_paths=("assets/local.bin",),
    )
    assert transaction.phase == "complete"
    assert read_tree(tree.root).tree_hash == tree.tree_hash
    assert tree.root.stat().st_ino == inode
    assert storage.snapshot(identity.local_id).document == document
    changed = document.with_description("Updated description")
    changed_baseline = replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint)
    storage.apply(identity, document.name, changed, changed_baseline,
                  expected_fingerprint=document.fingerprint)
    assert (tree.root / "assets/local.bin").read_bytes() == b"\x00\xfforiginal"
    assert storage.snapshot(identity.local_id).document == changed


def test_raw_tree_cas_includes_asset_and_chmod(tmp_path):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    (tree.root / "assets/local.bin").chmod(0o600)
    with pytest.raises(LinkError) as error:
        storage.adopt(identity, document.name, document, baseline,
                      expected_tree_hash=tree.tree_hash, snapshot=snapshot)
    assert error.value.code == "local-concurrent-edit"
    assert read_baseline(storage.config_dir, "skill", identity.local_id) is None


@pytest.mark.parametrize("phase", ["prepared", "old-renamed", "new-renamed", "content-committed"])
def test_crash_recovery_preserves_original_or_complete_result(tmp_path, phase):
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    changed = document.with_description("Updated description")

    def crash(current):
        if current == phase:
            raise RuntimeError("crash")
    with pytest.raises(RuntimeError, match="crash"):
        storage.adopt(
            identity, document.name, changed, baseline,
            expected_tree_hash=tree.tree_hash, snapshot=snapshot,
            carried_paths=("assets/local.bin",), checkpoint=crash,
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
    skill.write_text(skill.read_text().replace(
        "description:", "metadata:\n  local-rule: Keep every audit record\ndescription:",
    ))
    proposal = build_proposal(document.name, read_tree(tree.root), document,
                              resolver=None, allow_llm=False)
    assert proposal.document is not None
    assert "Keep every audit record" in proposal.document.to_json()["files"][0]["content"]
    local = skill.read_text().replace(document.description, "Unique local routing guidance")
    skill.write_text(local)
    proposal = build_proposal(document.name, read_tree(tree.root), document,
                              resolver=None, allow_llm=False)
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
    proposal = build_proposal(document.name, read_tree(tree.root), document,
                              resolver=None, allow_llm=False)
    assert proposal.state == "needs-choice"


def test_text_assets_never_reach_resolver(tmp_path):
    from local_operator.hub_sync.skill_document import SkillFile
    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    (tree.root / "assets/payload.txt").write_text("Unique local payload")
    remote = replace(
        document, files=document.files + (SkillFile("assets/payload.txt", "Hub payload"),),
    )

    class RejectCalls:
        def resolve(self, req):
            raise AssertionError("asset dispatched to model")

    proposal = build_proposal(document.name, read_tree(tree.root), remote, resolver=RejectCalls())
    assert proposal.state == "needs-choice"
    assert any(plan.path == "assets/payload.txt" for plan in proposal.unresolved)


def test_subscribe_then_adopt_existing_authored_folder(tmp_path):
    from local_operator.hub_sync import service
    from .test_skill_state import _rig, _subscribe, HUB_ID, ALIAS
    ctx, manager, remote, base, payload = _rig(tmp_path)
    subscribed = _subscribe(ctx, base)
    folder = manager.config_dir / "skills" / ALIAS
    for file in payload["files"]:
        path = folder / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file["content"])
    before = read_tree(folder).tree_hash
    arguments: dict[str, Any] = dict(
        hub_source=base, hub_id=HUB_ID, visibility="public", tenant_id=None, alias=ALIAS,
    )
    preview = service.install_skill(ctx, dry_run=True, **arguments)
    result = service.install_skill(
        ctx, confirm=True, adoption_receipt=preview["adoption_receipt"], **arguments,
    )
    assert result["reports"][0]["applied"]
    assert read_tree(folder).tree_hash == before
    record = read_baseline(manager.config_dir, "skill", subscribed["link"]["local_id"])
    assert record is not None and record.content_state == "present"
