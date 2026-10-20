"""Per-skill content state: subscribe without pull, unpull, and the check/apply arms.

A link's membership is its baseline + policy; ``content_state`` says whether the
managed copy exists. These tests drive the REAL services and the REAL
``ManagedSkillStorage`` on an isolated config root and assert persisted effects
(files, baseline, policy, status rows), never a call count of the subject's own
helpers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager, read_config_values
from local_operator.hub_sync import policies as pol
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import LinkError, Visibility
from local_operator.hub_sync.skill_storage import ManagedSkillStorage
from local_operator.providers.radient_credentials import configured_radient_base_url

from .test_skill_storage import _input

HUB_ID = "sk-1"
ALIAS = "example-skill"
REVISION = '"hub-skill-rev-1"'


class RemoteSkill:
    """The fetch seam: one skill DETAIL response (manifest + metadata envelope)."""

    def __init__(
        self,
        document: dict[str, Any],
        *,
        base: str,
        visibility: str = "public",
        tenant_id: str | None = None,
    ) -> None:
        self.document = document
        self.base_url = base
        self.visibility = visibility
        self.tenant_id = tenant_id
        self.error: Exception | None = None
        self.calls = 0

    def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        payload = dict(self.document)
        payload.update(
            {
                "revision": REVISION,
                "content_revision": 3,
                "sync_contract": {"revision_cas": 1, "skills": 1},
                "id": HUB_ID,
                "visibility": self.visibility,
                "tenant_id": self.tenant_id,
                "moderation": {"flagged": False},
            }
        )
        return payload


def _rig(tmp_path: Path, *, visibility: str = "public", tenant_id: str | None = None):
    root = tmp_path / "profile"
    manager = ConfigManager(root)
    base = configured_radient_base_url(manager)
    document = _input()[1].to_json()
    remote = RemoteSkill(document, base=base, visibility=visibility, tenant_id=tenant_id)
    ctx = svc.HubSyncContext(
        config_dir=root,
        config_manager=manager,
        client_for_tenant=lambda _tenant: remote,
    )
    return ctx, manager, remote, base, document


def _subscribe(
    ctx: svc.HubSyncContext,
    base: str,
    *,
    visibility: Visibility = "public",
    tenant_id: str | None = None,
    alias: str = ALIAS,
) -> dict[str, Any]:
    return svc.subscribe_skill(
        ctx,
        hub_source=base,
        hub_id=HUB_ID,
        visibility=visibility,
        tenant_id=tenant_id,
        alias=alias,
    )


def _record(root: Path) -> prov.BaselineRecord:
    records = [r for r in prov.iter_baselines(root) if r.kind == "skill"]
    assert len(records) == 1
    return records[0]


# -- provenance ----------------------------------------------------------------------


def test_a_record_without_content_state_reads_as_present() -> None:
    """Every pre-existing baseline has content, so the absent key MUST be present.

    Records written before this field existed are the overwhelming majority on a
    real machine; reading them as absent would hide every installed skill.
    """

    _, document, baseline = _input()
    raw = baseline.to_json()
    del raw["content_state"]
    parsed = prov.parse_baseline_record(raw, "skill", baseline.local_id)
    assert parsed is not None and parsed.content_state == "present"
    assert parsed.fields == document.to_json()


def test_an_unknown_content_state_is_an_unreadable_record() -> None:
    _, _, baseline = _input()
    raw = baseline.to_json()
    raw["content_state"] = "gone"
    assert prov.parse_baseline_record(raw, "skill", baseline.local_id) is None


# -- subscribe -----------------------------------------------------------------------


def test_subscribe_creates_an_absent_link_with_policy_and_no_files(tmp_path: Path) -> None:
    ctx, _manager, remote, base, document = _rig(tmp_path)

    result = _subscribe(ctx, base)

    row = result["link"]
    assert row["kind"] == "skill" and row["name"] == ALIAS
    assert row["content_state"] == "absent"
    assert row["local_path"] is None
    assert row["policy"]["subscribe"] is True and row["policy"]["auto_push"] is False
    assert row["eligibility"] == {"can_auto_push": False, "reason": "content-absent"}
    assert row["auto_will_apply"] is False and row["auto_will_push"] is False
    # The remote WAS read (identity validation is the point), but nothing landed.
    assert remote.calls == 1
    assert not (ctx.config_dir / "skills" / ALIAS).exists()

    record = _record(ctx.config_dir)
    assert record.content_state == "absent"
    assert record.identity is not None and record.identity.link_id == row["link_id"]
    assert record.recorded_by == "subscribe" and record.local_alias == ALIAS
    assert record.fields == document
    policy = pol.read_policy(read_config_values(ctx.config_dir) or {}, record.identity)
    assert policy.policy.subscribe is True and policy.policy.auto_push is False


def test_a_repeated_subscribe_replays_the_same_link(tmp_path: Path) -> None:
    ctx, _manager, _remote, base, _document = _rig(tmp_path)

    first = _subscribe(ctx, base)
    second = _subscribe(ctx, base)

    assert second["link"]["link_id"] == first["link"]["link_id"]
    assert len([r for r in prov.iter_baselines(ctx.config_dir) if r.kind == "skill"]) == 1
    assert not (ctx.config_dir / "skills" / ALIAS).exists()


# -- unpull --------------------------------------------------------------------------


def _present_link(tmp_path: Path):
    """A real installed skill link (the same write path a pull uses)."""

    from .test_skill_apply import _profile

    return _profile(tmp_path)


def test_unpull_keeps_the_link_policy_and_alias(tmp_path: Path) -> None:
    storage, _hub, ctx, identity, document = _present_link(tmp_path)
    values = read_config_values(storage.config_dir) or {}
    current = pol.read_policy(values, identity)
    before = pol.update_policy(
        ctx.config_manager, identity, expected_revision=current.revision, subscribe=False
    )
    assert pol.stored_link_policy(read_config_values(storage.config_dir) or {}, identity.link_id)

    result = svc.unpull_skill(ctx, link_id=identity.link_id)

    row = result["link"]
    assert row["content_state"] == "absent" and row["local_path"] is None
    assert not (storage.config_dir / "skills" / document.name).exists()
    # Membership survives: baseline, alias reservation and the EXACT policy entry.
    record = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert record is not None and record.content_state == "absent"
    assert record.identity == identity
    assert prov.alias_owner(storage.config_dir, "skill", document.name) == identity.local_id
    after = pol.stored_link_policy(read_config_values(storage.config_dir) or {}, identity.link_id)
    assert after is not None and after.to_json() == before.policy.to_json()
    assert row["policy"]["subscribe"] is False


def test_unpull_is_idempotent(tmp_path: Path) -> None:
    storage, _hub, ctx, identity, document = _present_link(tmp_path)

    first = svc.unpull_skill(ctx, link_id=identity.link_id)
    second = svc.unpull_skill(ctx, link_id=identity.link_id)

    assert first["link"]["content_state"] == "absent"
    assert second["link"]["content_state"] == "absent"
    assert second["link"]["link_id"] == first["link"]["link_id"]
    assert not (storage.config_dir / "skills" / document.name).exists()


def test_a_crash_between_the_rename_and_the_state_write_is_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash window: bytes moved aside, state still ``present``.

    The check that runs next must settle the link from the durable marker +
    tombstone and report a normal absent/available row -- never the corruption
    the missing directory would otherwise look like.
    """

    storage, _hub, ctx, identity, document = _present_link(tmp_path)
    original = ManagedSkillStorage._record_absent
    calls = {"n": 0}

    def crashing(self: ManagedSkillStorage, record: prov.BaselineRecord) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("power loss between rename and state write")
        original(self, record)

    monkeypatch.setattr(ManagedSkillStorage, "_record_absent", crashing)

    with pytest.raises(OSError):
        svc.unpull_skill(ctx, link_id=identity.link_id)

    marker = storage.unpull_marker(identity.local_id)
    assert marker.exists()
    stored = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert stored is not None and stored.content_state == "present"
    assert not (storage.config_dir / "skills" / document.name).exists()

    # The next CHECK settles it: recovery finishes the state write from evidence
    # rather than inventing absence from the missing directory.
    remote = RemoteSkill(document.to_json(), base="https://example.invalid/v1")
    ctx.client_for_tenant = lambda _tenant: remote
    checks = svc.check_items(ctx, kinds=("skill",))

    (check,) = checks
    assert check.verdict == "available" and check.reason == ""
    assert check.content_state == "absent" and check.local is None
    assert not marker.exists()
    settled = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert settled is not None and settled.content_state == "absent"
    # No tombstone outlives the settled transition.
    assert not list((storage.config_dir / "hub" / "backups").glob("*unpull-*"))


# -- check / apply -------------------------------------------------------------------


def test_an_explicit_pull_resumes_a_subscribed_link(tmp_path: Path) -> None:
    """Pull is the install a subscription waits for: it resumes, never duplicates."""

    ctx, _manager, _remote, base, _document = _rig(tmp_path)
    first = _subscribe(ctx, base)["link"]

    result = svc.install_skill(
        ctx,
        hub_source=base,
        hub_id=HUB_ID,
        visibility="public",
        tenant_id=None,
        alias=ALIAS,
    )

    (report,) = result["reports"]
    assert report["outcome"] == "merged" and report["applied"] is True
    assert result["link"]["link_id"] == first["link_id"]
    assert result["link"]["content_state"] == "present"
    assert (ctx.config_dir / "skills" / ALIAS / "SKILL.md").exists()
    records = [r for r in prov.iter_baselines(ctx.config_dir) if r.kind == "skill"]
    assert len(records) == 1 and records[0].content_state == "present"
    assert records[0].recorded_by == "pull" and records[0].lineage == "clean"


def test_recovery_aborts_an_unpull_that_never_renamed(tmp_path: Path) -> None:
    """Marker written, crash before the rename: the copy stays, no state is invented."""

    import json

    storage, _hub, ctx, identity, document = _present_link(tmp_path)
    marker = storage.unpull_marker(identity.local_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "operation": "a" * 32,
                "local_id": identity.local_id,
                "alias": document.name,
                "tombstone": f"skill-{identity.local_id}-unpull-{'a' * 32}",
            }
        )
    )

    storage.recover_unpull(local_id=identity.local_id)

    assert not marker.exists()
    assert (storage.config_dir / "skills" / document.name / "SKILL.md").exists()
    record = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert record is not None and record.content_state == "present"


def test_unpull_settles_a_crashed_pull_journal_first(tmp_path: Path) -> None:
    """Review round 1 nit: a prepared pull must not make unpull refuse."""

    storage, _hub, ctx, identity, document = _present_link(tmp_path)
    record = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert record is not None
    policy = pol.read_policy(read_config_values(storage.config_dir) or {}, identity)
    storage.journal.prepare(
        identity,
        "pull",
        before_fingerprint=storage.snapshot(identity.local_id).fingerprint,
        after_fingerprint="f" * 64,
        payload={
            "adapter": "managed-skill",
            "local_alias": document.name,
            "baseline": record.to_json(),
            "policy_revision": policy.revision,
            "automatic": False,
        },
    )

    row = svc.unpull_skill(ctx, link_id=identity.link_id)

    assert row["link"]["content_state"] == "absent"
    assert not storage.journal.pending(kind="skill", local_id=identity.local_id)


def test_a_stale_unpull_marker_cannot_mask_a_later_genuine_loss(tmp_path: Path) -> None:
    """Review round 1 MAJOR: the marker must be settled by the content write.

    An unpull can land rename + state write and die before deleting its marker.
    If the user re-pulls instead of re-clicking unpull, the old marker AND its
    tombstone used to survive; a later genuine loss then matched the crash case
    and was recorded as a deliberate absence. The pull must settle them first.
    """

    import json
    import shutil

    ctx, _manager, _remote, base, _document = _rig(tmp_path)
    _subscribe(ctx, base)
    svc.install_skill(
        ctx, hub_source=base, hub_id=HUB_ID, visibility="public", tenant_id=None, alias=ALIAS
    )
    record = _record(ctx.config_dir)
    assert record.identity is not None

    # Unpull completed, then "crashed" before deleting marker and tombstone.
    svc.unpull_skill(ctx, link_id=record.identity.link_id)
    marker = ManagedSkillStorage(ctx.config_dir).unpull_marker(record.local_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    operation = "b" * 32
    tombstone = ctx.config_dir / "hub" / "backups" / f"skill-{record.local_id}-unpull-{operation}"
    tombstone.mkdir(parents=True, exist_ok=True)
    (tombstone / "SKILL.md").write_text("the unpulled copy\n")
    marker.write_text(
        json.dumps(
            {
                "operation": operation,
                "local_id": record.local_id,
                "alias": ALIAS,
                "tombstone": tombstone.name,
            }
        )
    )

    # Re-pull, not re-unpull: the content write settles the stale intent.
    svc.install_skill(
        ctx, hub_source=base, hub_id=HUB_ID, visibility="public", tenant_id=None, alias=ALIAS
    )
    assert not marker.exists()
    assert not tombstone.exists()

    # Now the fresh copy is genuinely lost. No evidence says it was removed, so
    # it must stay the error it always was -- not become "absent".
    shutil.rmtree(ctx.config_dir / "skills" / ALIAS)
    (check,) = svc.check_items(ctx, kinds=("skill",))
    assert check.verdict == "unavailable" and check.reason == "local-concurrent-edit"
    stored = _record(ctx.config_dir)
    assert stored.content_state == "present"


def test_replace_local_on_an_absent_link_refuses(tmp_path: Path) -> None:
    """'Keep my side' has no side to keep: refuse by name, write nothing."""

    ctx, _manager, _remote, base, _document = _rig(tmp_path)
    _subscribe(ctx, base)
    checks = svc.check_items(ctx, kinds=("skill",))

    report = svc.apply_items(ctx, kind="skill", checks=checks, replace="local", allow_llm=False)

    (result,) = report.reports
    assert result.outcome == "refused" and result.error_class == "merge-refused"
    assert "no local copy" in result.message
    assert not (ctx.config_dir / "skills" / ALIAS).exists()
    assert _record(ctx.config_dir).content_state == "absent"


def test_kind_order_is_agent_then_team_then_skill_everywhere(tmp_path: Path) -> None:
    from local_operator.hub_sync.identity import LinkIdentity

    root = tmp_path / "profile"
    ctx = svc.HubSyncContext(
        config_dir=root,
        config_manager=ConfigManager(root),
        client_for_tenant=lambda _tenant: None,
    )
    base = "https://example.invalid/v1"
    fields = {
        "agent": {"instructions": "i", "description": "d"},
        "team": {
            "description": "d",
            "manager": "m",
            "members": [],
            "instructions": "",
            "project": "",
        },
        "skill": _input()[1].to_json(),
    }
    for kind in ("skill", "team", "agent"):
        local_id = f"{kind[0]}1"
        identity = LinkIdentity(kind, local_id, base, "public", None, f"h-{kind}")
        record = prov.make_record(kind, local_id, identity.hub_id, None, fields[kind], "pull")
        prov.write_baseline(
            root,
            replace(
                record,
                identity=identity,
                local_alias="zz-skill" if kind == "skill" else None,
                content_state="absent" if kind == "skill" else "present",
            ),
        )

    def fold(doc: dict[str, Any]) -> None:
        items = doc.setdefault("items", {})
        for kind in ("skill", "team", "agent"):
            items[f"{kind}:{kind[0]}1"] = {
                "kind": kind,
                "name": kind,
                "local_id": f"{kind[0]}1",
                "state": "available",
            }

    st.StatusStore(root).mutate(fold)
    assert [item["kind"] for item in svc.status_snapshot(ctx)["items"]] == [
        "agent",
        "team",
        "skill",
    ]
    assert [link["kind"] for link in svc.links_snapshot(ctx)["links"]] == [
        "agent",
        "team",
        "skill",
    ]


def test_a_check_on_an_absent_link_fetches_and_reports_available(tmp_path: Path) -> None:
    ctx, _manager, remote, base, _document = _rig(tmp_path)
    _subscribe(ctx, base)
    remote.calls = 0

    (check,) = svc.check_items(ctx, kinds=("skill",))

    assert remote.calls == 1  # it FETCHED instead of erroring on no directory
    assert check.verdict == "available"
    assert check.content_state == "absent"
    assert check.local is None and check.local_fp is None
    assert check.remote is not None and check.remote_fp is not None
    assert check.classification is None and check.reason == ""
    item = st.StatusStore(ctx.config_dir).load()["items"][f"skill:{check.local_id}"]
    assert item["state"] == "available" and item["error_class"] is None


def test_apply_on_an_absent_row_installs_wholesale(tmp_path: Path) -> None:
    ctx, _manager, _remote, base, document = _rig(tmp_path)
    _subscribe(ctx, base)

    # The explicit per-item apply is the install the subscription was waiting
    # for; it is the same shape the per-item route composes (auto is False).
    checks = svc.check_items(ctx, kinds=("skill",))
    report = svc.apply_items(ctx, kind="skill", checks=checks, allow_llm=False)

    (result,) = report.reports
    assert result.outcome == "merged" and result.applied is True
    skill_dir = ctx.config_dir / "skills" / ALIAS
    assert (skill_dir / "SKILL.md").exists()
    record = _record(ctx.config_dir)
    assert record.content_state == "present" and record.recorded_by == "merge-pull"
    assert record.fields == document
    row = svc.links_snapshot(ctx)["links"][0]
    assert row["content_state"] == "present"
    assert row["local_path"] == str(skill_dir)


@pytest.mark.asyncio
async def test_the_runner_never_auto_installs_an_absent_subscribed_skill(
    tmp_path: Path,
) -> None:
    from local_operator.hub_sync.runner import HubSyncRunner

    ctx, manager, _remote, base, _document = _rig(tmp_path)
    _subscribe(ctx, base)
    runner = HubSyncRunner(config_manager=manager, build_ctx=lambda: ctx, startup_delay=0)

    first = await runner.tick(reason="startup")  # first-run grace, by design
    second = await runner.tick(reason="timer")

    assert first.applied == 0 and second.applied == 0
    assert second.checked >= 1 and second.available >= 1
    assert not (ctx.config_dir / "skills" / ALIAS).exists()
    item = st.StatusStore(ctx.config_dir).load()["items"][f"skill:{ctx_skill_id(ctx)}"]
    assert item["state"] == "available"


def ctx_skill_id(ctx: svc.HubSyncContext) -> str:
    record = _record(ctx.config_dir)
    return record.local_id


# -- updates surface -----------------------------------------------------------------


def test_skill_rows_reach_the_updates_status_and_can_be_retried(tmp_path: Path) -> None:
    ctx, _manager, remote, base, _document = _rig(tmp_path)
    _subscribe(ctx, base)
    remote.error = RuntimeError("404 Not Found")
    remote.error.status_code = 404  # type: ignore[attr-defined]

    (failed,) = svc.check_items(ctx, kinds=("skill",))
    assert failed.verdict == "unavailable" and failed.reason == "hub-item-missing"

    snapshot = svc.status_snapshot(ctx)
    (row,) = [item for item in snapshot["items"] if item["kind"] == "skill"]
    assert row["state"] == "failed" and row["error_class"] == "hub-item-missing"
    assert row["next_retry_at"] is not None

    # The retry the route composes: clear the backoff and check again. The
    # successful check retires the 404 class and reports the remote as
    # available; the apply step is UNATTENDED, so it skips the install and the
    # row stays available for an explicit pull.
    svc.clear_retry(ctx, "skill", ALIAS)
    remote.error = None
    checks = svc.check_items(ctx, kinds=("skill",))
    report = svc.apply_items(ctx, kind="skill", checks=checks, auto=True, dry_run=True)
    assert report.reports[0].outcome == "skipped"
    assert report.reports[0].skipped_reason == "install-requires-pull"

    (row,) = [
        item
        for item in svc.status_snapshot(ctx)["items"]
        if item["kind"] == "skill" and item["name"] == ALIAS
    ]
    assert row["error_class"] is None and row["state"] == "available"
    assert not (ctx.config_dir / "skills" / ALIAS).exists()


# -- scope ---------------------------------------------------------------------------


def test_an_org_skill_is_refused_by_the_public_scope_with_one_plain_404(
    tmp_path: Path,
) -> None:
    """A public request must not be able to tell an org row from a missing one."""

    ctx, _manager, _remote, base, _document = _rig(tmp_path, visibility="org", tenant_id="org-1")

    with pytest.raises(LinkError) as public:
        _subscribe(ctx, base)
    with pytest.raises(LinkError) as wrong_org:
        _subscribe(ctx, base, visibility="org", tenant_id="org-2")

    assert public.value.code == "skill-not-found"
    assert wrong_org.value.code == "skill-not-found"
    assert str(public.value) == str(wrong_org.value)
    assert not [r for r in prov.iter_baselines(ctx.config_dir) if r.kind == "skill"]

    # A proper org subscription exists, but the typed agent/team list path never
    # returns it: the public list surface cannot leak an org link.
    _subscribe(ctx, base, visibility="org", tenant_id="org-1")
    assert svc.links_snapshot(ctx, kinds=("agent", "team"))["links"] == []
    (skill_row,) = svc.links_snapshot(ctx, kinds=("skill",))["links"]
    assert skill_row["visibility"] == "org" and skill_row["tenant_id"] == "org-1"


def test_update_all_does_not_install_an_absent_subscribed_skill(tmp_path: Path) -> None:
    """The batch action is not an install: only an explicit pull writes a copy."""

    ctx, _manager, _remote, base, _document = _rig(tmp_path)
    _subscribe(ctx, base)

    report = svc.apply_items(ctx, kind="skill", all_available=True, allow_llm=False)

    (result,) = report.reports
    assert result.outcome == "skipped" and result.skipped_reason == "content-absent"
    assert not (ctx.config_dir / "skills" / ALIAS).exists()
    assert _record(ctx.config_dir).content_state == "absent"


def test_a_lost_directory_is_still_the_error_it_was(tmp_path: Path) -> None:
    """Absence is never INFERRED: only an explicit state write may record it."""

    import shutil

    storage, _hub, ctx, identity, document = _present_link(tmp_path)
    shutil.rmtree(storage.config_dir / "skills" / document.name)
    remote = RemoteSkill(document.to_json(), base="https://example.invalid/v1")
    ctx.client_for_tenant = lambda _tenant: remote

    (check,) = svc.check_items(ctx, kinds=("skill",))

    assert check.verdict == "unavailable" and check.reason == "local-concurrent-edit"
    stored = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert stored is not None and stored.content_state == "present"

    # Unpull refuses it too: it cannot prove the copy was ours to remove, so it
    # does not manufacture the absent state the check just refused to infer.
    with pytest.raises(LinkError) as refused:
        svc.unpull_skill(ctx, link_id=identity.link_id)
    assert refused.value.code == "local-concurrent-edit"
    stored = prov.read_baseline(storage.config_dir, "skill", identity.local_id)
    assert stored is not None and stored.content_state == "present"
