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
    """A fake hub serving one skill DETAIL response (the check/pull seam).

    The payload carries the METADATA envelope a real response has beside the
    manifest (``revision``, ``sync_contract``, ids -- agent-server@bdd03ab,
    internal/responses/skills.go HubSkillResponse), so every test here also
    exercises the tolerant-envelope/strict-manifest split (round-2 QN1).
    """

    def __init__(self, document: SkillDocument, *, revision: str = '"hub-skill-rev-1"') -> None:
        self.document = document
        self.revision = revision

    def client(self) -> Any:
        return self

    def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
        payload = dict(self.document.to_json())
        payload.update(
            {
                "revision": self.revision,
                "content_revision": 3,
                "sync_contract": {"revision_cas": 1, "skills": 1},
                "id": "sk-1",
                "visibility": "org",
                "tenant_id": "t1",
                "moderation": {"flagged": False},
            }
        )
        return payload


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


def test_the_check_tolerates_the_metadata_envelope_and_captures_the_revision(
    tmp_path: Path,
) -> None:
    """QN1/QN2: the DETAIL envelope is tolerated, its revision becomes the check's.

    Before the split, one metadata key in the response made ``from_json`` raise
    out of ``check_items`` and take every other kind's checks down with it; the
    revision the push needs is that response's ``revision``.
    """

    from local_operator.hub_sync import check as chk

    storage, hub, ctx, identity, _doc = _profile(tmp_path)
    checks = chk.check_hub_skills(
        config_dir=storage.config_dir, client_for_tenant=lambda _t: hub.client()
    )
    (c,) = checks
    assert c.verdict == "up-to-date"
    assert c.remote_revision == hub.revision
    assert c.sync_contract == {"revision_cas": 1, "skills": 1}


def test_a_broken_manifest_is_a_named_unavailable_not_an_exception(tmp_path: Path) -> None:
    """QN1: a malformed remote manifest stays ONE named row and never raises."""

    import dataclasses

    storage, hub, ctx, identity, _doc = _profile(tmp_path)
    broken = hub.get_skill("x")
    broken.pop("description")  # a MANIFEST field is missing; metadata still rides

    class BrokenHub:
        api_key = "k"
        base_url = "https://hub.test/v1"

        def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
            return dict(broken)

    checks = svc.check_items(
        dataclasses.replace(ctx, client_for_tenant=lambda _t: BrokenHub()), kinds=("skill",)
    )
    (c,) = checks
    assert c.verdict == "unavailable" and c.reason == "invalid-remote-manifest"


def test_a_public_skill_check_asks_for_no_credential(tmp_path: Path) -> None:
    """QN3: the anonymous client must not demand a key for a public skill row."""

    import inspect

    from local_operator.clients.radient import RadientClient
    from local_operator.hub_sync import check as chk

    storage, hub, ctx, identity, _doc = _profile(tmp_path)
    flag = [
        p
        for p in inspect.signature(RadientClient.get_skill).parameters
        if p not in ("self", "skill_id", "timeout")
    ][0]

    seen: list[dict[str, Any]] = []

    class Anon:
        api_key = None
        base_url = "https://hub.test/v1"

        def get_skill(self, skill_id: str, **kw: Any) -> dict[str, Any]:
            seen.append(kw)
            return hub.get_skill(skill_id)

    checks = chk.check_hub_skills(
        config_dir=storage.config_dir, client_for_tenant=lambda _t: Anon()
    )
    (c,) = checks
    assert c.verdict == "up-to-date"
    assert seen and seen[0].get(flag) is False  # no credential demanded or sent


def test_a_remote_added_reference_file_lands_through_the_service(tmp_path: Path) -> None:
    """QN4: a file ADDED upstream is an add, not an eternal needs-review."""

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    v2_json = v1.to_json()
    v2_json["files"] = v2_json["files"] + [{"path": "references/extra.md", "content": "extra\n"}]
    hub.document = SkillDocument.from_json(v2_json)

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "merged" and report.reports[0].applied, report.reports[
        0
    ].message
    assert {f.path for f in _disk(storage, identity).files} >= {
        "references/extra.md",
        "SKILL.md",
    }
    assert _baseline(storage.config_dir, identity.local_id).fields == hub.document.to_json()


def test_a_locally_added_reference_file_survives_a_remote_change(tmp_path: Path) -> None:
    """QN4: a local add is not refused as "an edit against a removal"."""

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    local_json = v1.to_json()
    local_json["files"] = local_json["files"] + [
        {"path": "references/local.md", "content": "mine\n"}
    ]
    local = SkillDocument.from_json(local_json)
    storage.apply(
        identity,
        local.name,
        local,
        _baseline(storage.config_dir, identity.local_id),
        expected_fingerprint=_disk(storage, identity).fingerprint,
    )
    hub.document = _input("# New body\n")[1]

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome in ("merged", "unchanged"), report.reports[0].message
    files = {f.path: f.content for f in _disk(storage, identity).files}
    # The write path normalizes trailing whitespace (the engine's seg.norm), so
    # the surviving local add is compared by content, not by newline count.
    assert "references/local.md" in files and files["references/local.md"].rstrip("\n") == "mine"
    assert "New body" in files["SKILL.md"]


def test_both_sides_adding_the_same_path_differently_holds_for_review(tmp_path: Path) -> None:
    """QN4: add/add with DIFFERENT text is a real conflict, silently nothing."""

    storage, hub, ctx, identity, v1 = _profile(tmp_path)
    local_json = v1.to_json()
    local_json["files"] = local_json["files"] + [{"path": "references/x.md", "content": "LOCAL\n"}]
    local = SkillDocument.from_json(local_json)
    storage.apply(
        identity,
        local.name,
        local,
        _baseline(storage.config_dir, identity.local_id),
        expected_fingerprint=_disk(storage, identity).fingerprint,
    )
    remote_json = v1.to_json()
    remote_json["files"] = remote_json["files"] + [
        {"path": "references/x.md", "content": "REMOTE\n"}
    ]
    hub.document = SkillDocument.from_json(remote_json)

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "needs-review", report.reports[0].message
    files = {f.path: f.content for f in _disk(storage, identity).files}
    assert files["references/x.md"] == "LOCAL\n"  # nothing written
    assert _baseline(storage.config_dir, identity.local_id).fields == v1.to_json()


def test_a_fast_forward_pull_seeds_the_outbound_receipt(tmp_path: Path) -> None:
    """Seeding ruling (A)a: integrating the remote agrees the pair (A, P, revision)."""

    storage, hub, ctx, identity, _v1 = _profile(tmp_path)
    v2 = _input("# New body\n")[1]
    hub.document = v2

    report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False)

    assert report.reports[0].applied
    stored = _baseline(storage.config_dir, identity.local_id)
    assert stored.outbound is not None
    assert stored.outbound.remote_revision == hub.revision
    assert stored.outbound.accepted_fields == v2.to_json()


def test_the_fake_refuses_unknown_keys_and_expired_pins_like_the_server() -> None:
    """R19: the fake enforces the server's CLOSED top-level key sets per family
    on preview and commit, and refuses an expired pin."""

    import httpx

    from tests.unit.hub_sync.fake_hub import FakeHub

    hub = FakeHub()
    valid = {
        "document_version": 1,
        "name": "s",
        "description": "d",
        "files": [{"path": "SKILL.md", "content": "---\nname: s\ndescription: d\n---\n\nbody\n"}],
    }
    headers = {"Authorization": "Bearer test", "If-Match": hub.revision}

    def preview(url: str, document: object) -> httpx.Response:
        return hub.handler(
            httpx.Request(
                "PUT",
                url,
                headers=headers,
                json={"preview_version": 1, "document": document},
            )
        )

    # Unknown key named BY the decoder, on preview and on commit, per family.
    bad = preview("https://hub.test/v1/skills/publish/preview", {**valid, "sneaky": 1})
    assert bad.status_code == 422 and "sneaky" in bad.text
    bad_agent = preview(
        "https://hub.test/v1/agents/h1/publish/preview",
        {"instructions": "x", "model": "gpt-5"},  # model is REFUSED by the agent decoder
    )
    assert bad_agent.status_code == 422 and "model" in bad_agent.text

    ok = preview("https://hub.test/v1/skills/publish/preview", valid)
    assert ok.status_code == 200
    token = ok.json()["result"]["pin"]["token"]
    commit_bad = hub.handler(
        httpx.Request(
            "PUT",
            "https://hub.test/v1/skills/publish",
            headers={**headers, "X-Radient-Preview-Token": token},
            json={**valid, "sneaky": 1},
        )
    )
    assert commit_bad.status_code == 422 and "sneaky" in commit_bad.text

    # Pin TTL: an elapsed pin is refused, never served.
    hub.pin_ttl_seconds = -1.0
    ok2 = preview("https://hub.test/v1/skills/publish/preview", valid)
    token2 = ok2.json()["result"]["pin"]["token"]
    commit_expired = hub.handler(
        httpx.Request(
            "PUT",
            "https://hub.test/v1/skills/publish",
            headers={**headers, "X-Radient-Preview-Token": token2},
            json=valid,
        )
    )
    assert commit_expired.status_code == 422
