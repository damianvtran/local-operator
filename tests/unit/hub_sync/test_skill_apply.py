"""Skill pull/apply writes through the managed transaction; B moves only with it.

The round-1 QA repro: ``service.apply_one`` fell to the generic path for skills,
merged nothing, advanced the baseline and settled 'up-to-date' without touching
the skill. These tests drive the REAL path (check -> apply -> managed write) and
assert PERSISTED effects: the bytes on disk, the baseline record, the status row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage

from .test_skill_storage import _input


class SkillHub:
    """A fake hub serving one skill manifest (the check/pull network seam)."""

    def __init__(self, document: SkillDocument) -> None:
        self.document = document

    def client(self) -> Any:
        return self

    def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
        return self.document.to_json()


def _profile(tmp_path: Path):
    """A managed skill v1 on disk + a ctx whose client serves the CURRENT remote."""

    identity, document, baseline = _input()
    root = tmp_path / "profile"
    storage = ManagedSkillStorage(root)
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)
    hub = SkillHub(document)
    ctx = svc.HubSyncContext(
        config_dir=root,
        config_manager=ConfigManager(root),
        client_for_tenant=lambda _t: hub.client(),
    )
    return storage, hub, ctx, identity, document


def _disk(storage: ManagedSkillStorage, identity: Any) -> SkillDocument:
    return storage.snapshot(identity.local_id).document


def _baseline(root: Path, local_id: str):
    record = prov.read_baseline(root, "skill", local_id)
    assert record is not None
    return record


def test_a_remote_only_change_fast_forwards_the_managed_skill(tmp_path: Path) -> None:
    """Q3: the remote v2 must LAND (files + baseline), not just move the baseline."""

    storage, hub, ctx, identity, _v1 = _profile(tmp_path)
    v2 = _input("# New body\n")[1]
    hub.document = v2

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "merged" and report.reports[0].applied
    assert "New body" in _disk(storage, identity).files[0].content
    stored = _baseline(storage.config_dir, identity.local_id)
    assert stored.fields == v2.to_json()  # B := the exact integrated remote
    item = st.StatusStore(storage.config_dir).load()["items"][f"skill:{identity.local_id}"]
    assert item["state"] == "applied"


def test_an_edit_against_a_removal_holds_for_review_and_moves_nothing(tmp_path: Path) -> None:
    """The genuinely unresolvable case: one side removed the file the other edited.

    (Two same-atom prose edits are NOT this class: the engine's sanctioned
    additive fallback combines them deterministically and that merge IS writable.)
    """

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    local = SkillDocument(
        v1.name, v1.description, tuple(f for f in v1.files if f.path == "SKILL.md")
    )
    storage.apply(
        identity,
        local.name,
        local,
        _baseline(storage.config_dir, identity.local_id),
        expected_fingerprint=_disk(storage, identity).fingerprint,
    )
    remote_json = v1.to_json()
    remote_json["files"] = [
        {
            "path": f.path,
            # a SAME-atom edit on the file the local side removed (an append-only
            # change would be disjoint and legitimately merge)
            "content": f.content.replace("never run", "never load"),
        }
        for f in v1.files
    ]
    hub.document = SkillDocument.from_json(remote_json)  # remote EDITS inert.py

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "needs-review", report.reports[0].message
    # Nothing written, and B stays on the last INTEGRATED document.
    assert "remote edit" not in "".join(f.content for f in _disk(storage, identity).files)
    stored = _baseline(storage.config_dir, identity.local_id)
    assert stored.fields == v1.to_json()


def test_an_unknown_origin_skill_never_advances_its_baseline(tmp_path: Path) -> None:
    """The invariant: without a writer identity a skill apply must refuse."""

    from local_operator.hub_sync.check import ItemCheck

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    # Simulate a legacy unknown-origin record by stripping the identity from the
    # stored JSON directly: write_baseline deliberately REFUSES to drop one.
    import json

    path = prov.baseline_path(storage.config_dir, "skill", identity.local_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("identity", None)
    path.write_text(json.dumps(raw), encoding="utf-8")
    hub.document = _input("# New body\n")[1]
    check = ItemCheck(
        "skill",
        identity.local_id,
        identity.local_id,
        identity.hub_id,
        None,
        "available",
        local=v1.to_json(),
        remote=hub.document.to_json(),
        base=v1.to_json(),
    )

    report = svc.apply_one(ctx, check, auto=True, allow_llm=False)

    assert report.error_class == "baseline-unknown", report
    stored = _baseline(storage.config_dir, identity.local_id)
    assert stored.identity is None  # nothing re-created the writer identity
    assert "# Body" in _disk(storage, identity).files[0].content  # untouched


def test_a_clean_two_sided_merge_applies_and_names_stay_split(tmp_path: Path) -> None:
    """Disjoint edits merge; the baseline keeps the SOURCE name, the disk the alias."""

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    remote = _input()  # remote edits description only
    manifest = remote[1].to_json()
    manifest = {**manifest, "description": "Example skill v2"}
    manifest["files"][0]["content"] = manifest["files"][0]["content"].replace(
        "description: Example skill\n", "description: Example skill v2\n"
    )
    hub.document = SkillDocument.from_json(manifest)  # remote description change

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "merged"
    assert _disk(storage, identity).description == "Example skill v2"
    stored = _baseline(storage.config_dir, identity.local_id)
    assert stored.fields["name"] == "example-skill"  # source name in ancestry
    assert stored.local_alias == "example-skill"


def test_removing_a_reference_file_propagates(tmp_path: Path) -> None:
    """A deterministic one-sided removal is honoured (no ratio-only refusal)."""

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    remote_keep = [f for f in v1.files if f.path == "SKILL.md"]
    hub.document = SkillDocument(v1.name, v1.description, tuple(remote_keep))

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "merged", report.reports[0].message
    paths = {file.path for file in _disk(storage, identity).files}
    assert "scripts/inert.py" not in paths and "SKILL.md" in paths
    stored = _baseline(storage.config_dir, identity.local_id)
    assert stored.fields == hub.document.to_json()  # B := the removal decision
