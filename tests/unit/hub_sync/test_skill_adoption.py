"""Authored skill adoption exercises the service and the lossless merge boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from local_operator.hub_sync import service as svc
from local_operator.hub_sync import skill_adoption as adoption
from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.merge import ConflictProposal, ConflictRequest
from local_operator.hub_sync.skill_document import SkillDocument

from .test_skill_state import ALIAS, HUB_ID, _record, _rig


def _authored(root: Path, document: dict[str, Any]) -> Path:
    folder = root / "skills" / ALIAS
    for entry in document["files"]:
        path = folder / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["content"], encoding="utf-8")
    return folder


def _install(ctx: svc.HubSyncContext, base: str, **kwargs: Any) -> dict[str, Any]:
    return svc.install_skill(
        ctx,
        hub_source=base,
        hub_id=HUB_ID,
        visibility="public",
        tenant_id=None,
        alias=ALIAS,
        **kwargs,
    )


def test_equal_authored_alias_is_adopted_by_real_install(tmp_path: Path) -> None:
    ctx, manager, _remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    preview = _install(ctx, base, dry_run=True)
    assert preview["adoption"]["state"] == "equal"
    result = _install(ctx, base, confirm=True, adoption_receipt=preview["adoption_receipt"])

    assert result["reports"][0]["applied"] is True
    assert _record(manager.config_dir).fields == SkillDocument.from_json(document).to_json()
    assert {
        p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    } == before


def test_dry_run_authored_collision_previews_instead_of_would_install(tmp_path: Path) -> None:
    ctx, manager, _remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    (folder / "SKILL.md").write_text(
        (folder / "SKILL.md").read_text() + "\nKeep the municipality POC guidance.\n"
    )
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    result = _install(ctx, base, dry_run=True)

    assert result["reports"][0]["outcome"] != "would-install", result
    assert result["adoption"]["state"] in {"ready", "needs-choice"}
    assert result["adoption_receipt"]
    assert result["snapshot"]["tree_hash"]
    assert {
        p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    } == before
    assert not list((manager.config_dir / "hub-sync").glob("**/baseline.json"))


class FaithfulModel:
    """Keep both passages so validator acceptance, not a mocked merge, is tested."""

    def __init__(self) -> None:
        self.requests: list[ConflictRequest] = []

    def resolve(self, req: ConflictRequest) -> ConflictProposal:
        self.requests.append(req)
        return ConflictProposal(
            req.local + "\n\n" + req.remote,
            covers=("l1", "r1"),
            model="fixture-faithful-model",
        )


def _municipal(tmp_path: Path, *, changed: bool = True):
    description = "Municipal procurement guidance"
    frontmatter = f"---\nname: {ALIAS}\ndescription: {description}\n---\n"
    local_quality = (
        "Verify every municipality procurement deadline against the official notice. "
        "Preserve the unique POC clerk confirmation before the pilot."
    )
    hub_quality = (
        "Verify every municipality procurement deadline against the official notice. "
        "Preserve the unique procurement audit evidence before the pilot."
    )
    local_body = (
        "# Municipal procurement\n\n"
        "Read [requirements](requirements.md) and [rubric](rubric.md).\n\n"
        "## Quality\n" + (local_quality if changed else hub_quality) + "\n"
    )
    hub_body = (
        "# Municipal procurement\n\n"
        "Read [requirements](references/requirements.md) and "
        "[rubric](references/rubric.md).\n\n"
        "## Quality\n" + hub_quality + "\n"
    )
    requirements = "# Requirements\nKeep a dated audit trail for each municipality.\n"
    rubric = "# Rubric\nScore the evidence, not assumptions about eligibility.\n"
    document = {
        "document_version": 1,
        "name": ALIAS,
        "description": description,
        "files": [
            {"path": "SKILL.md", "content": frontmatter + hub_body},
            {"path": "references/requirements.md", "content": requirements},
            {"path": "references/rubric.md", "content": rubric},
        ],
    }
    folder = _authored(
        tmp_path,
        {
            "files": [
                {"path": "SKILL.md", "content": frontmatter + local_body},
                {"path": "requirements.md", "content": requirements},
                {"path": "rubric.md", "content": rubric},
            ]
        },
    )
    return folder, SkillDocument.from_json(document), local_quality, hub_quality


def _bytes(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def test_municipal_merge_preserves_both_guidance_and_relocates_without_model_paths(
    tmp_path: Path,
) -> None:
    folder, remote, local_quality, hub_quality = _municipal(tmp_path)
    (folder / "local-notes.md").write_text("Keep the POC escalation owner on the checklist.\n")
    (folder / "assets").mkdir()
    binary = b"\x00\xffPRIVATE-BINARY-NOT-FOR-THE-MODEL\x89PNG"
    (folder / "assets" / "seal.png").write_bytes(binary)
    before = _bytes(folder)
    model = FaithfulModel()
    tree = adoption.read_tree(folder)

    proposal = adoption.build_proposal(ALIAS, tree, remote, resolver=model)

    assert proposal.state == "ready", proposal.to_json()
    assert proposal.document is not None
    files = {entry.path: entry.content for entry in proposal.document.files}
    for guidance in (local_quality, hub_quality):
        for sentence in guidance.split(". "):
            assert sentence in files["SKILL.md"]
    assert "[requirements](references/requirements.md)" in files["SKILL.md"]
    assert "[rubric](references/rubric.md)" in files["SKILL.md"]
    assert "(requirements.md)" not in files["SKILL.md"]
    assert "(rubric.md)" not in files["SKILL.md"]
    assert files["references/local-notes.md"] == before["local-notes.md"].decode()
    assert {p.local_path: p.path for p in proposal.files if p.action == "relocated"} == {
        "requirements.md": "references/requirements.md",
        "rubric.md": "references/rubric.md",
        "local-notes.md": "references/local-notes.md",
    }
    assert [(p.path, p.action) for p in proposal.files if p.path == "assets/seal.png"] == [
        ("assets/seal.png", "carried")
    ]
    assert "assets/seal.png" not in files
    # Unknown-baseline additions can be retained mechanically; the model seam
    # must not receive attachments even if a future paragraph needs resolution.
    assert all(request.field == "SKILL.md" for request in model.requests)
    assert "PRIVATE-BINARY" not in repr(model.requests)
    assert _bytes(folder) == before
    assert proposal.local_tree_hash == tree.tree_hash
    assert {row["path"] for row in proposal.local_manifest} == set(before)


def test_normalized_equal_municipal_relocation_needs_no_model(tmp_path: Path) -> None:
    folder, remote, _local, _hub = _municipal(tmp_path, changed=False)
    model = FaithfulModel()
    before = _bytes(folder)

    proposal = adoption.build_proposal(ALIAS, adoption.read_tree(folder), remote, resolver=model)

    assert proposal.state == "equal", proposal.to_json()
    assert proposal.document == remote
    assert model.requests == []
    assert _bytes(folder) == before


@pytest.mark.parametrize("changed_side", ["local", "hub"])
def test_service_refuses_content_changed_after_preview(tmp_path: Path, changed_side: str) -> None:
    ctx, manager, remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    preview = _install(ctx, base, dry_run=True)
    if changed_side == "local":
        path = folder / "SKILL.md"
        path.write_text(path.read_text() + "\nUnreviewed local instructions.\n")
    else:
        remote.document = {
            **document,
            "files": [
                (
                    {**entry, "content": entry["content"] + "\nUnreviewed hub instructions.\n"}
                    if entry["path"] == "SKILL.md"
                    else entry
                )
                for entry in document["files"]
            ],
        }
    before = _bytes(folder)

    with pytest.raises(LinkError) as error:
        _install(ctx, base, confirm=True, adoption_receipt=preview["adoption_receipt"])

    assert error.value.code == "adoption-preview-changed"
    assert _bytes(folder) == before


def test_service_requires_review_even_when_equal(tmp_path: Path) -> None:
    ctx, manager, _remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    before = _bytes(folder)

    with pytest.raises(LinkError) as error:
        _install(ctx, base, confirm=True)

    assert error.value.code == "alias-occupied"
    assert _bytes(folder) == before


def test_relocation_destination_collision_preserves_both_local_files(tmp_path: Path) -> None:
    folder, remote, _local, _hub = _municipal(tmp_path, changed=False)
    (folder / "references").mkdir()
    unique = "# Existing local requirements\nKeep signed POC consent before outreach.\n"
    (folder / "references" / "requirements.md").write_text(unique)
    before = _bytes(folder)
    model = FaithfulModel()

    proposal = adoption.build_proposal(ALIAS, adoption.read_tree(folder), remote, resolver=model)

    # Two distinct local files cannot both occupy the Hub destination. This
    # must stop for an explicit choice, never select a source by traversal order.
    assert proposal.state == "needs-choice", proposal.to_json()
    assert proposal.document is None
    assert any("collides" in row.reason for row in proposal.files)
    assert _bytes(folder) == before


def test_service_municipal_merge_applies_reviewed_text_and_preserves_binary(tmp_path: Path) -> None:
    ctx, manager, remote, base, _document = _rig(tmp_path)
    folder, document, local_quality, hub_quality = _municipal(manager.config_dir)
    remote.document = document.to_json()
    (folder / "assets").mkdir()
    binary = b"\x00\xffPRIVATE-POC-SEAL"
    (folder / "assets" / "seal.png").write_bytes(binary)
    (folder / "local-notes.md").write_text("Keep the municipal escalation owner.\n")
    before = _bytes(folder)
    model = FaithfulModel()
    ctx.resolver = model

    preview = _install(ctx, base, dry_run=True)
    assert preview["adoption"]["state"] == "ready", preview
    assert _bytes(folder) == before
    snapshot = Path(preview["snapshot"]["path"])
    assert snapshot.is_dir()
    assert _bytes(snapshot / "content") == before
    calls = len(model.requests)
    result = _install(ctx, base, confirm=True, adoption_receipt=preview["adoption_receipt"])

    assert result["reports"][0]["applied"] is True
    body = (folder / "SKILL.md").read_text()
    for guidance in (local_quality, hub_quality):
        for sentence in guidance.split(". "):
            assert sentence in body
    assert (folder / "assets" / "seal.png").read_bytes() == binary
    assert (folder / "references" / "local-notes.md").read_text() == before[
        "local-notes.md"
    ].decode()
    assert (folder / "references" / "requirements.md").read_bytes() == before["requirements.md"]
    assert len(model.requests) == calls
    assert _record(manager.config_dir).fields == document.to_json()


def test_differing_description_uses_validated_model_proposal(tmp_path: Path) -> None:
    folder, remote, _local, _hub = _municipal(tmp_path, changed=False)
    path = folder / "SKILL.md"
    path.write_text(
        path.read_text().replace(
            "description: Municipal procurement guidance",
            "description: Municipal procurement guidance with clerk POC verification",
        )
    )
    model = FaithfulModel()

    proposal = adoption.build_proposal(ALIAS, adoption.read_tree(folder), remote, resolver=model)

    assert proposal.state == "ready", proposal.to_json()
    assert proposal.document is not None
    assert "clerk POC verification" in proposal.document.description
    assert model.requests
    assert all(request.field == "description" for request in model.requests)


def test_disabling_model_requires_explicit_fallback(tmp_path: Path) -> None:
    folder, remote, _local, _hub = _municipal(tmp_path)
    model = FaithfulModel()
    tree = adoption.read_tree(folder)

    proposal = adoption.build_proposal(ALIAS, tree, remote, resolver=model, allow_llm=False)

    assert proposal.state == "needs-choice", proposal.to_json()
    assert proposal.strategy is None
    assert proposal.document is None
    assert "SKILL.md" in [plan.path for plan in proposal.unresolved]
    assert model.requests == []


@pytest.mark.parametrize("strategy", ["keep-local", "take-hub"])
def test_explicit_fallback_resolves_only_the_chosen_guidance(
    tmp_path: Path, strategy: Literal["keep-local", "take-hub"]
) -> None:
    folder, remote, local_quality, hub_quality = _municipal(tmp_path)
    model = FaithfulModel()
    before = _bytes(folder)

    proposal = adoption.build_proposal(
        ALIAS,
        adoption.read_tree(folder),
        remote,
        resolver=model,
        allow_llm=False,
        strategy=strategy,
    )

    assert proposal.state == "ready", proposal.to_json()
    assert proposal.document is not None
    body = next(entry.content for entry in proposal.document.files if entry.path == "SKILL.md")
    assert (local_quality in body) is (strategy == "keep-local")
    assert (hub_quality in body) is (strategy == "take-hub")
    assert model.requests == []
    assert _bytes(folder) == before


# --- Relocated siblings referenced from INSIDE references/ (operator's real case) ---
#
# The real ``pergamon-municipal-officials`` folder keeps requirements.md and
# rubric.md at the root, while its references/ files mention them as bare inline
# code paths (lines copied verbatim below). Read from references/, the bare name
# resolves to references/<name> -- exactly where relocation puts the file. Those
# links were dangling locally and become CORRECT after the move, so the merge
# must accept them verbatim instead of refusing "ambiguous inline filepath".
# The original fixtures only referenced the moved files from the root SKILL.md,
# which is why the shipped 0.96.0 dead-ended on the operator's tree.

_ENRICHMENT_LINE = (
    "| `dob_coverage` | The run-level floor is **99%** of rows carrying a `dob_value`, "
    "stated or inferred (`requirements.md` R20). A row that cannot reach a stated date "
    "must be pushed to an inference rather than left empty, and an inference that stays "
    "too weak to publish is recorded in `notes` with the exact blocker. |\n"
)
_RECORD_LINE = "| `confidence` | `high` / `medium` / `low`, defined as in `rubric.md`. |\n"


def _municipal_with_sibling_refs(tmp_path: Path, *, changed: bool = True):
    folder, remote, local_quality, hub_quality = _municipal(tmp_path, changed=changed)
    (folder / "references").mkdir()
    (folder / "references" / "enrichment-contract.md").write_text(
        "# Enrichment contract\n\n| Field | Rule |\n| --- | --- |\n" + _ENRICHMENT_LINE
    )
    (folder / "references" / "record-contract.md").write_text(
        "# Record contract\n\n| Field | Rule |\n| --- | --- |\n" + _RECORD_LINE
    )
    data = remote.to_json()
    data["files"] = list(data["files"]) + [
        {
            "path": "references/enrichment-contract.md",
            "content": "# Enrichment contract\n\n| Field | Rule |\n| --- | --- |\n"
            + _ENRICHMENT_LINE,
        },
        {
            "path": "references/record-contract.md",
            "content": "# Record contract\n\n| Field | Rule |\n| --- | --- |\n" + _RECORD_LINE,
        },
    ]
    return folder, SkillDocument.from_json(data), local_quality, hub_quality


@pytest.mark.parametrize("strategy", ["merge", "keep-local", "take-hub"])
def test_sibling_refs_inside_references_never_dead_end(tmp_path: Path, strategy: str) -> None:
    folder, remote, _local, _hub = _municipal_with_sibling_refs(tmp_path)
    before = _bytes(folder)

    proposal = adoption.build_proposal(
        ALIAS, adoption.read_tree(folder), remote, resolver=FaithfulModel(), strategy=strategy
    )

    assert proposal.state == "ready", proposal.to_json()
    assert proposal.document is not None
    files = {entry.path: entry.content for entry in proposal.document.files}
    # Left verbatim: the bare sibling name is correct at its new home.
    assert _RECORD_LINE in files["references/record-contract.md"]
    assert _ENRICHMENT_LINE in files["references/enrichment-contract.md"]
    assert _bytes(folder) == before


def test_sibling_refs_equal_case_adopts_without_model(tmp_path: Path) -> None:
    folder, remote, _local, _hub = _municipal_with_sibling_refs(tmp_path, changed=False)
    model = FaithfulModel()

    proposal = adoption.build_proposal(ALIAS, adoption.read_tree(folder), remote, resolver=model)

    assert proposal.state == "equal", proposal.to_json()
    assert model.requests == []


def test_inline_path_to_relocated_basename_elsewhere_still_refuses(tmp_path: Path) -> None:
    """Scope guard: only the relocation's own destination is accepted.

    From ``references/sub/`` the bare name resolves to ``references/sub/rubric.md``:
    it neither exists locally nor is where rubric.md moves, so it is still
    genuinely ambiguous and the merge keeps asking for a choice.
    """
    folder, remote, _local, _hub = _municipal(tmp_path)
    (folder / "references" / "sub").mkdir(parents=True)
    (folder / "references" / "sub" / "guide.md").write_text("Scored as in `rubric.md`.\n")

    with pytest.raises(adoption.AdoptionError) as caught:
        adoption.build_proposal(ALIAS, adoption.read_tree(folder), remote, resolver=FaithfulModel())
    assert caught.value.code == "adoption-needs-choice"
