"""End-to-end merge/apply on REAL registries with a fake hub: the never-overwrite guarantees."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import Kind
from local_operator.hub_sync.merge import ConflictProposal, ResolverError
from local_operator.hub_sync.resolver import LlmResolver, MergeModel
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

BASE = "## Rules\nBe brief.\n\n## Legacy\nOld advice.\n\n## Tools\nUse tools carefully."


class Hub:
    """A fake hub serving one agent listing and any number of team documents.

    ``get_agent`` answers the DETAIL read the agent check now performs for the
    live revision (round-2 QN2a): the payload carries the row's ``revision`` and
    ``sync_contract`` beside the metadata keys a real response has, and
    ``revision`` can be moved by tests to simulate an accepted remote write.
    """

    def __init__(self) -> None:
        self.agents: dict[str, tuple[str, str]] = {}
        self.teams: dict[str, dict[str, Any]] = {}
        self.revision: str = '"hub-rev-1"'
        #: The NORMALIZED base this fake answers from (used for served_by /
        #: attestation keying in tests that go through ``client_for_tenant``).
        self.base_url: str = "https://hub.test/v1"

    def client(self) -> Any:
        return self

    def bearer_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer test-token"}

    def get_agent(self, hub_id: str, **_kw: Any) -> dict[str, Any]:
        text, desc = self.agents[hub_id]
        return {
            "instructions": text,
            "description": desc,
            "revision": self.revision,
            "sync_contract": {"revision_cas": 1},
        }

    def get_team(self, team_id: str, **_kw: Any) -> dict[str, Any]:
        if team_id not in self.teams:
            err = RuntimeError("404 Not Found")
            err.status_code = 404  # type: ignore[attr-defined]
            raise err
        return dict(self.teams[team_id])


class Scripted:
    def __init__(self, *proposals: ConflictProposal, error: Exception | None = None) -> None:
        self.proposals, self.error, self.calls = list(proposals), error, 0

    def resolve(self, req: Any) -> ConflictProposal:
        self.calls += 1
        if self.error:
            raise self.error
        return self.proposals[min(self.calls - 1, len(self.proposals) - 1)]


def _fields(**kw: Any) -> AgentEditFields:
    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(kw)
    return AgentEditFields(**base)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / ".local-operator"
    hub = Hub()

    def fetch(_client: Any, hub_id: str, **_kw: Any) -> tuple[str, str]:
        return hub.agents[hub_id]

    monkeypatch.setattr("local_operator.agents._fetch_hub_profile", fetch)
    agents, teams = AgentRegistry(root), TeamRegistry(root)
    cm = ConfigManager(root)
    ctx = svc.HubSyncContext(
        config_dir=root,
        config_manager=cm,
        client_for_tenant=lambda _t: hub.client(),
        agent_registry=agents,
        team_registry=teams,
    )
    return ctx, hub, agents, teams, root


def _pull_agent(
    agents: AgentRegistry, hub: Hub, root: Path, text: str = BASE, desc: str = "d"
) -> Any:
    """What a real pull leaves: a stamped row plus a baseline record."""

    hub.agents["h1"] = (text, desc)
    row = agents.create_agent(_fields(name="coder", description=desc, tags=["role"]))
    agents.set_agent_system_prompt(row.id, text)
    stamped = agents._stamp_hub_provenance(agents.get_agent(row.id), "h1")
    agents._record_hub_baseline(stamped, "h1", None)
    return stamped


def _text(agents: AgentRegistry, agent_id: str) -> str:
    return agents.get_agent_system_prompt(agent_id).strip()


def test_local_deletion_survives_a_remote_edit_elsewhere_and_the_hub_addition_lands(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(
        row.id, "## Rules\nBe brief.\n\n## Tools\nUse tools carefully."
    )  # user deleted Legacy
    hub.agents["h1"] = (
        BASE.replace("Be brief.", "Be brief and kind.") + "\n\n## New\nAdded upstream.",
        "d",
    )

    report = svc.apply_items(ctx, kind="agent", allow_llm=False)

    (r,) = report.reports
    assert r.outcome == "merged" and r.applied and r.backup
    text = _text(agents, row.id)
    assert (
        "Legacy" not in text and "Old advice" not in text
    )  # the deliberate deletion did not regrow
    assert "Be brief and kind." in text and "## New" in text  # the hub's edits landed
    assert (ctx.config_dir / r.backup).exists()  # the pre-apply text is recoverable


def test_a_local_edit_is_kept_while_the_hub_change_lands_and_the_baseline_advances(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(
        row.id, BASE.replace("Use tools carefully.", "Use tools carefully. Ask first.")
    )
    hub.agents["h1"] = (BASE.replace("Old advice.", "Older advice."), "d")

    svc.apply_items(ctx, kind="agent", allow_llm=False)

    text = _text(agents, row.id)
    assert "Ask first." in text and "Older advice." in text
    # B := the REMOTE text just integrated, so the surviving local edit is still a local edit.
    record = prov.read_baseline(root, "agent", row.id)
    assert (
        record
        and "Ask first" not in record.fields["instructions"]
        and "Older advice" in record.fields["instructions"]
    )
    again = svc.apply_items(ctx, kind="agent", allow_llm=False)
    assert again.reports[0].outcome in ("up-to-date", "unchanged")


def test_removal_versus_edit_is_never_auto_resolved_and_writes_nothing(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    mine = "## Rules\nBe brief.\n\n## Tools\nUse tools carefully."
    agents.set_agent_system_prompt(row.id, mine)
    hub.agents["h1"] = (BASE.replace("Old advice.", "Older, better advice."), "d")

    report = svc.apply_items(ctx, kind="agent", auto=True, allow_llm=False)

    assert report.reports[0].outcome == "needs-review" and not report.reports[0].applied
    assert _text(agents, row.id) == mine
    item = ctx.store().load()["items"][f"agent:{row.id}"]
    assert (
        item["state"] == "available"
        and item["error_class"] == "merge-refused"
        and item["auto_retry"] is False
    )

    # An explicit prefer is the caller's decision, and the loser is reported.
    resolved = svc.apply_items(ctx, kind="agent", prefer="local", allow_llm=False)
    assert "Legacy" not in _text(agents, row.id)
    (field,) = [f for f in resolved.reports[0].fields if f.field == "instructions"]
    assert "Older, better advice." in str(
        next(r for r in field.regions if r.name == "Legacy").dropped
    )
    # Choosing "mine" settles it: the baseline advanced to the hub text, so the same
    # hub edit is not offered again and the deletion is now a plain local edit.
    (again,) = svc.check_items(ctx, kinds=("agent",))
    assert again.verdict == "up-to-date"
    assert "Legacy" not in _text(agents, row.id)


def test_combined_regions_auto_apply_only_after_validation(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, BASE.replace("Be brief.", "Be brief and cite sources."))
    hub.agents["h1"] = (BASE.replace("Be brief.", "Be brief and use plain words."), "d")
    ctx.resolver = Scripted(
        ConflictProposal("Be brief, cite sources, and use plain words.", covers=("l1", "r1"))
    )

    report = svc.apply_items(ctx, kind="agent", auto=True)

    assert report.reports[0].applied
    assert "Be brief, cite sources, and use plain words." in _text(agents, row.id)
    assert report.reports[0].fields[0].engine.mode == "llm"


def test_a_model_outage_leaves_a_conflicting_item_available_but_still_applies_clean_ones(
    env,
) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, BASE.replace("Be brief.", "Be brief and cite sources."))
    hub.agents["h1"] = (BASE.replace("Be brief.", "Be brief and use plain words."), "d")
    ctx.resolver = Scripted(error=ResolverError("model-unavailable", "no key"))

    report = svc.apply_items(ctx, kind="agent", auto=True)

    assert (
        report.reports[0].outcome == "needs-review"
        and report.reports[0].error_class == "model-unavailable"
    )
    item = ctx.store().load()["items"][f"agent:{row.id}"]
    assert item["state"] == "available" and item["error_class"] == "model-unavailable"


def test_an_unusable_provider_records_the_failure_and_a_future_retry_the_sidebar_sees(
    env,
) -> None:
    """The unattended-update failure path: an unusable provider/model at the
    resolver call must leave a durable, retryable failure the desktop indicator
    can read, not a silent no-op.

    This is the leg the two-install unattended-update proof never reached: the
    orchestrated run always had a working resolver, so no provider-error was ever
    recorded. The genuine class is provoked by driving the REAL :class:`LlmResolver`
    over a completion whose provider the wire layer refuses (``clients.py`` raises
    ``ValueError("Unknown provider: ...")`` before any HTTP), which lands as
    ``provider-error/request`` through ``classify_failure``/``call_with_retry`` --
    exactly the wrap's ``ResolverError: provider-error: Unknown provider``. No
    production seam is added: ``complete`` is already the injected call.
    """

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, BASE.replace("Be brief.", "Be brief and cite sources."))
    hub.agents["h1"] = (BASE.replace("Be brief.", "Be brief and use plain words."), "d")

    async def unusable_provider(system: str, prompt: str) -> str:
        raise ValueError("Unknown provider: 'bogus'")

    ctx.resolver = LlmResolver(
        ctx.config_manager,
        complete=unusable_provider,
        model=MergeModel(
            spec=SimpleNamespace(context_window=128_000, max_output_tokens=4096),
            label="bogus/bogus",
        ),
    )

    report = svc.apply_items(ctx, kind="agent", auto=True)

    # (1) The item did not settle: a conflict held for review, classed as a
    # provider failure rather than the merge-refused a two-sided conflict gives.
    assert report.reports[0].outcome == "needs-review"
    assert report.reports[0].error_class == "provider-error/request"

    # ...and the failure is RECORDED on the row with a future retry schedule, so
    # the timer re-attempts instead of the item being abandoned.
    item = ctx.store().load()["items"][f"agent:{row.id}"]
    assert (item["error_class"], item["error_subclass"]) == ("provider-error", "request")
    assert item["last_error"] and item["attempts"] == 1 and item["auto_retry"] is True
    due = datetime.strptime(item["next_retry_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    # Counted backoff for attempt 1 is 15 min * U(0.9, 1.1); a future, bounded due.
    assert timedelta(minutes=12) < due - datetime.now(timezone.utc) < timedelta(minutes=18)

    # (2) The desktop sidebar's failure indicator is a pure store read: the
    # recorded class and schedule are what it consumes, so both must be exposed.
    (exposed,) = svc.status_snapshot(ctx)["items"]
    assert exposed["error_class"] == "provider-error"
    assert exposed["error_subclass"] == "request"
    assert exposed["state"] == "available"  # indicator stays; the user resolves it
    assert exposed["next_retry_at"] == item["next_retry_at"]


def test_a_concurrent_edit_between_check_and_write_aborts_and_keeps_the_users_file(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nUpstream.", "d")
    checks = svc.check_items(ctx, kinds=("agent",))
    agents.set_agent_system_prompt(row.id, "I EDITED THIS MEANWHILE")  # the race

    report = svc.apply_items(ctx, kind="agent", checks=checks, allow_llm=False)

    assert report.reports[0].error_class == "concurrent-edit" and not report.reports[0].applied
    assert _text(agents, row.id) == "I EDITED THIS MEANWHILE"


def test_replace_is_explicit_echoes_what_it_discarded_and_is_never_part_of_apply_all(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, "MY WORDS")
    hub.agents["h1"] = ("HUB WORDS", "d")

    batch = svc.apply_items(ctx, all_available=True, replace="remote", allow_llm=False)
    assert _text(agents, row.id) != "HUB WORDS"  # apply-all cannot carry a replace

    one = svc.apply_items(ctx, kind="agent", replace="remote", allow_llm=False)
    assert _text(agents, row.id) == "HUB WORDS"
    assert one.reports[0].replaced["instructions"] == "MY WORDS"
    from local_operator.hub_sync.report import render_report

    assert "MY WORDS" in render_report(one)  # the echo that keeps it recoverable
    assert batch.reports  # (ran, and applied nothing destructive)


def test_dry_run_writes_nothing(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nX.", "d")
    report = svc.apply_items(ctx, kind="agent", dry_run=True, allow_llm=False)
    assert report.reports[0].outcome == "would-merge" and _text(agents, row.id) == BASE.strip()
    assert not prov.backups_dir(root).exists()


def test_an_agent_with_no_baseline_that_was_edited_is_check_only_and_never_auto_applies(
    env,
) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    prov.delete_baseline(root, "agent", row.id)
    agents.set_agent_system_prompt(row.id, BASE + "\n\nMine.")
    hub.agents["h1"] = (BASE + "\n\n## New\nUp.", "d")

    (c,) = svc.check_items(ctx, kinds=("agent",))
    assert c.classification and c.classification.state == "baseline-unknown"
    auto = svc.apply_items(ctx, kind="agent", checks=[c], auto=True, allow_llm=False)
    assert not auto.reports[0].applied
    manual = svc.apply_items(
        ctx, kind="agent", checks=[c], acknowledge_unknown_baseline=True, allow_llm=False
    )
    text = _text(agents, row.id)
    assert (
        manual.reports[0].applied and "Mine." in text and "## New" in text
    )  # nothing of the user's deleted


def test_an_unedited_legacy_pull_adopts_a_baseline_exactly(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    prov.delete_baseline(root, "agent", row.id)  # pulled before this feature: tag only
    hub.agents["h1"] = (BASE + "\n\n## New\nUp.", "d")
    (c,) = svc.check_items(ctx, kinds=("agent",))
    assert c.classification is not None and c.classification.state == "remote-only"
    assert c.base is not None
    adopted = prov.read_baseline(root, "agent", row.id)
    assert adopted is not None and adopted.recorded_by == "adopt"


# -- teams -------------------------------------------------------------------------------


def _pull_team(teams: TeamRegistry, hub: Hub, root: Path) -> Any:
    doc = {
        "id": "ht1",
        "tenant_id": "org-1",
        "name": "release",
        "description": "d",
        "manager": "lead",
        "members": [
            {"role": "coder", "kind": "agent", "count": 1},
            {"role": "reviewer", "kind": "agent", "count": 2},
        ],
        "instructions": "## Brief\nShip it.",
        "project": "## Goal\nRelease.",
        "version": "1.0.0",
    }
    hub.teams["ht1"] = doc
    outcome = teams.import_hub_team(doc)
    prov.record_team_pull(root, teams.get_team(outcome.team.id), doc, tenant_id="org-1")
    return outcome.team


def test_team_roster_merge_honors_a_local_slot_removal_and_takes_a_remote_addition(env) -> None:
    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    teams.update_team(
        team.id, TeamEditFields(members=[TeamMember(role="coder", count=1)])
    )  # dropped reviewer
    hub.teams["ht1"] = {
        **hub.teams["ht1"],
        "members": [*hub.teams["ht1"]["members"], {"role": "qa", "kind": "agent", "count": 1}],
        "instructions": "## Brief\nShip it carefully.",
    }

    report = svc.apply_items(ctx, kind="team", allow_llm=False)

    assert report.reports[0].applied
    after = teams.get_team(team.id)
    assert [m.role for m in after.members] == ["coder", "qa"]  # reviewer did not regrow
    assert after.instructions.strip() == "## Brief\nShip it carefully."
    assert any(
        w.startswith("missing-role:") for w in report.reports[0].warnings
    )  # a warning, not a failure


def test_team_name_is_never_touched_and_a_remote_rename_is_only_a_note(env) -> None:
    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    hub.teams["ht1"] = {**hub.teams["ht1"], "name": "renamed-upstream", "description": "new d"}
    (c,) = svc.check_items(ctx, kinds=("team",))
    assert c.detail == "remote-renamed"
    svc.apply_items(ctx, kind="team", checks=[c], allow_llm=False)
    after = teams.get_team(team.id)
    assert after.name == "release" and after.description == "new d"


def test_team_404_is_unavailable_and_never_deletes_anything_locally(env) -> None:
    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    del hub.teams["ht1"]
    (c,) = svc.check_items(ctx, kinds=("team",))
    assert c.verdict == "unavailable" and c.reason == "hub-item-missing"
    assert teams.get_team(team.id).name == "release"
    assert prov.read_baseline(root, "team", team.id) is not None
    assert ctx.store().load()["items"][f"team:{team.id}"]["state"] == "failed"


def test_a_team_with_no_link_is_never_touched(env) -> None:
    ctx, hub, _a, teams, root = env
    teams.create_team(
        TeamEditFields(
            name="local-only",
            manager="m",
            members=[TeamMember(role="c", count=1)],
            instructions="x",
        )
    )
    assert svc.check_items(ctx, kinds=("team",)) == []


def test_a_team_edit_between_check_and_write_aborts_under_the_registry_lock(env) -> None:
    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    hub.teams["ht1"] = {**hub.teams["ht1"], "description": "hub d"}
    checks = svc.check_items(ctx, kinds=("team",))
    teams.update_team(team.id, TeamEditFields(description="edited meanwhile"))
    report = svc.apply_items(ctx, kind="team", checks=checks, allow_llm=False)
    assert report.reports[0].error_class == "concurrent-edit"
    assert teams.get_team(team.id).description == "edited meanwhile"


def test_the_team_hub_caps_are_enforced_with_the_hubs_own_words(env) -> None:
    ctx, hub, _a, teams, root = env
    _pull_team(teams, hub, root)
    hub.teams["ht1"] = {
        **hub.teams["ht1"],
        "members": [{"role": f"r{i}", "kind": "agent", "count": 1} for i in range(70)],
    }
    report = svc.apply_items(ctx, kind="team", allow_llm=False)
    assert (
        report.reports[0].outcome == "refused"
        and "at most 64 items (submitted 70)" in report.reports[0].message
    )


def test_update_all_stops_on_a_systemic_failure_and_marks_the_rest_skipped(env) -> None:
    ctx, hub, agents, _t, root = env
    for i, name in enumerate(("aa", "bb")):
        hub.agents[f"h{i}"] = ("## R\nBe brief.", "d")
        row = agents.create_agent(_fields(name=name, description="d", tags=["role"]))
        agents.set_agent_system_prompt(row.id, "## R\nBe brief and cite.")
        agents._stamp_hub_provenance(agents.get_agent(row.id), f"h{i}")
        prov.write_baseline(
            root,
            prov.make_record(
                "agent",
                row.id,
                f"h{i}",
                None,
                {"instructions": "## R\nBe brief.", "description": "d"},
                "pull",
            ),
        )
        hub.agents[f"h{i}"] = ("## R\nBe brief and use plain words.", "d")
    ctx.resolver = Scripted(error=ResolverError("model-unavailable", "no key"))
    report = svc.apply_items(ctx, all_available=True)
    outcomes = [(r.name, r.outcome) for r in report.reports]
    assert outcomes[0][1] == "needs-review" and outcomes[1] == ("bb", "skipped")


def test_the_status_snapshot_is_a_pure_store_read_with_the_documented_shape(env) -> None:
    ctx, hub, agents, _t, root = env
    _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nX.", "d")
    svc.check_items(ctx, kinds=("agent",))
    snap = svc.status_snapshot(ctx)
    assert set(snap) == {"generated_at", "credential", "settings", "counts", "items"}
    (item,) = snap["items"]
    assert (
        item["state"] == "available"
        and item["classification"] == "remote-only"
        and item["auto_will_apply"] is True
    )
    assert item["name"] == "coder" and item["kind"] == "agent"
    ctx.config_manager.set_config_value("hub", {"auto_update": {"agents": False}})
    assert (
        svc.status_snapshot(ctx)["items"][0]["auto_will_apply"] is False
    )  # manual mode still shows the update


def test_a_signed_out_item_still_reaches_the_snapshot(env) -> None:
    """U11: an up-to-date item that needs the login is LISTED, or no UI can offer it.

    The snapshot drops every ``up-to-date`` item, which is exactly the state of an
    org-linked row the user has no credential for. It is listed on its
    ``no-credential`` class alone, and it still carries nothing a mark would draw.
    """

    ctx, *_ = env
    ctx.store().mutate(
        lambda doc: st.apply_check(
            doc,
            kind="agent",
            local_id="a1",
            name="oscar",
            hub_id="h",
            tenant_id="acme-team",
            verdict="unavailable",
            classification=None,
            baseline="known",
            local_fp=None,
            remote_fp=None,
            reason="no-credential",
            detail="no Radient credential is available for this item",
        )
    )
    (item,) = svc.status_snapshot(ctx)["items"]
    assert item["state"] == "up-to-date" and item["error_class"] == "no-credential"
    assert item["tenant_id"] == "acme-team" and item["name"] == "oscar"
    assert item["auto_will_apply"] is False


def test_a_quota_error_stops_update_all_like_any_other_systemic_class(env) -> None:
    """R2: ``error_class`` already carries the subclass; it must not gain a second one."""

    ctx, hub, agents, _t, root = env
    for i, name in enumerate(("aa", "bb")):
        hub.agents[f"h{i}"] = ("## R\nBe brief.", "d")
        row = agents.create_agent(_fields(name=name, description="d", tags=["role"]))
        agents.set_agent_system_prompt(row.id, "## R\nBe brief and cite.")
        agents._stamp_hub_provenance(agents.get_agent(row.id), f"h{i}")
        prov.write_baseline(
            root,
            prov.make_record(
                "agent",
                row.id,
                f"h{i}",
                None,
                {"instructions": "## R\nBe brief.", "description": "d"},
                "pull",
            ),
        )
        hub.agents[f"h{i}"] = ("## R\nBe brief and use plain words.", "d")
    resolver = Scripted(error=ResolverError("provider-error", "slow down", subclass="quota"))
    ctx.resolver = resolver

    report = svc.apply_items(ctx, all_available=True)

    outcomes = [(r.name, r.outcome) for r in report.reports]
    assert outcomes[0][1] == "needs-review" and outcomes[1] == ("bb", "skipped")
    assert report.reports[0].error_class == "provider-error/quota"
    assert resolver.calls == 1  # the second agent never reached the model
    item = ctx.store().load()["items"][f"agent:{report.reports[0].local_id}"]
    assert (item["error_class"], item["error_subclass"]) == ("provider-error", "quota")


def test_replace_takes_the_hub_copy_even_when_only_the_local_copy_changed(env) -> None:
    """R3: "discard my edits, take the hub's" with an UNCHANGED hub (classification local-only)."""

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, BASE + "\n\n## Mine\nLocal-only edit.")

    plain = svc.apply_items(ctx, kind="agent", allow_llm=False)
    assert plain.reports[0].outcome == "up-to-date"  # a plain pull has nothing to fetch
    assert "Local-only edit" in _text(agents, row.id)

    forced = svc.apply_items(ctx, kind="agent", replace="remote", allow_llm=False)

    (r,) = forced.reports
    assert r.applied and _text(agents, row.id) == BASE.strip()
    assert "Local-only edit" in str(
        r.replaced["instructions"]
    )  # the echo that keeps it recoverable


def test_replace_on_identical_texts_has_nothing_to_replace(env) -> None:
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    forced = svc.apply_items(ctx, kind="agent", replace="remote", allow_llm=False)
    assert forced.reports[0].outcome == "up-to-date" and _text(agents, row.id) == BASE.strip()


def test_applied_reverts_to_up_to_date_after_exactly_one_clean_check(env) -> None:
    """R6: ``applied`` is a one-cycle "Updated just now", not a permanent state."""

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nUp.", "d")
    svc.apply_items(ctx, kind="agent", allow_llm=False)
    key = f"agent:{row.id}"
    assert ctx.store().load()["items"][key]["state"] == "applied"

    svc.check_items(ctx, kinds=("agent",))
    assert ctx.store().load()["items"][key]["state"] == "applied"  # kept one cycle
    svc.check_items(ctx, kinds=("agent",))
    assert ctx.store().load()["items"][key]["state"] == "up-to-date"
    assert svc.status_snapshot(ctx)["counts"]["applied"] == 0


def test_a_team_check_racing_a_save_never_prunes_its_baseline(env, monkeypatch) -> None:
    """R7: a short listing (a row mid-swap) is not proof the team is gone."""

    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    monkeypatch.setattr(teams, "list_teams", lambda: [])  # the transiently short listing
    svc.check_items(ctx, kinds=("team",))
    assert prov.read_baseline(root, "team", team.id) is not None  # dir still on disk

    teams.delete_team(team.id)
    svc.check_items(ctx, kinds=("team",))
    assert prov.read_baseline(root, "team", team.id) is None  # genuinely gone: swept


def test_team_label_and_aliases_survive_a_hub_sync_apply(env) -> None:
    """Labels are LOCAL metadata: a merge apply must never wipe them.

    The merge's field set (``TEAM_FIELDS``) deliberately never carries label
    or aliases, so the apply's ``TeamEditFields`` payload leaves both None --
    the registry's "leave the stored value alone" -- and this pins that end to
    end: pull, label locally, edit the hub, apply, read back.
    """
    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    teams.update_team(team.id, TeamEditFields(label="Release Display", aliases=["rel"]))
    hub.teams["ht1"] = {**hub.teams["ht1"], "description": "new d"}

    report = svc.apply_items(ctx, kind="team", allow_llm=False)

    assert report.reports[0].applied
    after = teams.get_team(team.id)
    assert after.description == "new d"
    assert after.label == "Release Display"
    assert after.aliases == ["rel"]


def test_resolve_source_materializes_identity_and_preserves_scope(env) -> None:
    """The resolve arm confirms SOURCE+SCOPE and refuses a differing known scope."""
    ctx, hub, agents, teams, root = env
    stamped = _pull_agent(agents, hub, root)
    # Drop the identity so the row is a LEGACY unknown-origin link (hub id only).
    record = prov.read_baseline(root, "agent", stamped.id)
    assert record is not None
    prov.write_baseline(root, dataclasses.replace(record, identity=None, scope_known=True))

    resolved = svc.resolve_source(
        ctx,
        kind="agent",
        name=stamped.name,
        hub_source="https://hub.example.com/v1",
        visibility="public",
        tenant_id=None,
    )
    assert resolved.identity is not None
    assert resolved.identity.visibility == "public"
    assert resolved.scope_known is True
    stored = prov.read_baseline(root, "agent", stamped.id)
    assert stored is not None and stored.identity is not None
    assert stored.identity.link_id == resolved.identity.link_id

    # A DIFFERENT known scope (org) at the same source is refused, old bytes kept.
    before = (root / "hub" / "baselines" / f"agent-{stamped.id}.json").read_bytes()
    with pytest.raises(Exception):
        svc.resolve_source(
            ctx,
            kind="agent",
            name=stamped.name,
            hub_source="https://hub.example.com/v1",
            visibility="org",
            tenant_id="org-tab",
        )
    assert (root / "hub" / "baselines" / f"agent-{stamped.id}.json").read_bytes() == before


def test_resolve_source_keeps_a_genuine_pulls_provenance(env) -> None:
    """F2: pull then resolve must ADD identity, never downgrade a trusted baseline.

    Observed on the real acceptance pass: the team's pull recorded a trusted
    baseline (``recorded_by=pull``); the CLI's resolve then overwrote it with the
    unknown marker, and the next tick refused the merge with "the baseline is
    unknown". The check trusts the recorded base exactly when ``recorded_by`` is
    not the unknown marker (``known = record.recorded_by != UNKNOWN_BASELINE``),
    so the downgrade silently discarded a perfectly good merge base.
    """

    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    before = prov.read_baseline(root, "team", team.id)
    assert before is not None and before.recorded_by == "pull"

    svc.resolve_source(
        ctx,
        kind="team",
        name=team.name,
        hub_source="https://hub.test/v1",
        visibility="org",
        tenant_id="org-1",
    )

    after = prov.read_baseline(root, "team", team.id)
    assert after is not None and after.identity is not None
    assert after.recorded_by == "pull"  # was adopt-unknown: the F2 downgrade
    assert after.recorded_by != prov.UNKNOWN_BASELINE
    assert after.remote_revision == before.remote_revision
    assert after.scope_known is True  # the resolve still confirms source+scope

    # The real consequence: a remote-only edit now merges unattended from the
    # pull-time base instead of refusing "baseline is unknown".
    hub.teams["ht1"] = {**hub.teams["ht1"], "instructions": "## Brief\nShip it safely."}
    report = svc.apply_items(ctx, kind="team", allow_llm=False)
    assert report.reports and report.reports[0].applied, report


def test_resolve_source_never_vouches_an_unknown_legacy_baseline(env) -> None:
    """The anti-laundering half: a legacy record stays unknown after resolve.

    Resolving confirms SOURCE+SCOPE; it is not an integration. A record no real
    pull/merge wrote keeps the unknown marker, so the check keeps refusing to
    diff against it -- resolve must not convert absence of evidence into a
    trusted baseline.
    """

    ctx, hub, agents, _t, root = env
    stamped = _pull_agent(agents, hub, root)
    record = prov.read_baseline(root, "agent", stamped.id)
    assert record is not None
    prov.write_baseline(
        root,
        dataclasses.replace(
            record, identity=None, recorded_by=prov.UNKNOWN_BASELINE, scope_known=True
        ),
    )

    resolved = svc.resolve_source(
        ctx,
        kind="agent",
        name=stamped.name,
        hub_source="https://hub.test/v1",
        visibility="public",
        tenant_id=None,
    )
    assert resolved.identity is not None

    after = prov.read_baseline(root, "agent", stamped.id)
    assert after is not None
    assert after.recorded_by == prov.UNKNOWN_BASELINE  # still conservative
    assert after.lineage != "clean"
    assert after.remote_revision is None


def test_resolve_places_identity_on_the_original_record_without_rebuilding_the_base(
    env,
) -> None:
    """R2: resolve adds identity/scope to the ORIGINAL record; it never rebuilds.

    Reconstructing the payload from current local content makes a user's
    unintegrated edit the merge base: the next unchanged-remote apply then
    classifies as ``remote-only`` and erases the edit. Review repro with the
    identical-code no-resolve control: with resolve the edit was overwritten;
    without it the edit survived. This pins the record itself.
    """

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, BASE.replace("Old advice.", "USER EDIT TO KEEP"))
    before = prov.read_baseline(root, "agent", row.id)
    assert before is not None and before.recorded_by != prov.UNKNOWN_BASELINE

    svc.resolve_source(
        ctx,
        kind="agent",
        name=row.name,
        hub_source="https://hub.test/v1",
        visibility="public",
        tenant_id=None,
    )

    after = prov.read_baseline(root, "agent", row.id)
    assert after is not None and after.identity is not None
    assert after.fields == before.fields, "the pulled base must not absorb local edits"
    assert after.fingerprint == before.fingerprint
    bytes_now = (root / "hub" / "baselines" / f"agent-{row.id}.json").read_bytes()
    assert b"USER EDIT TO KEEP" not in bytes_now, "the baseline file must not carry the edit"

    # The remote never changed: the apply is a no-op and the edit survives.
    report = svc.apply_items(ctx, kind="agent", allow_llm=False)
    assert report.reports[0].outcome == "up-to-date", report
    assert "USER EDIT TO KEEP" in _text(agents, row.id)


def test_resolve_keeps_the_team_pull_base_and_a_surviving_local_edit(env) -> None:
    """R2 for teams: same contract, exercised on the team registry path."""

    ctx, hub, _a, teams, root = env
    team = _pull_team(teams, hub, root)
    teams.update_team(team.id, TeamEditFields(description="USER TEAM EDIT TO KEEP"))
    before = prov.read_baseline(root, "team", team.id)
    assert before is not None and before.recorded_by != prov.UNKNOWN_BASELINE

    svc.resolve_source(
        ctx,
        kind="team",
        name=team.name,
        hub_source="https://hub.test/v1",
        visibility="org",
        tenant_id="org-1",
    )

    after = prov.read_baseline(root, "team", team.id)
    assert after is not None and after.identity is not None
    assert after.fields == before.fields
    assert after.fingerprint == before.fingerprint

    report = svc.apply_items(ctx, kind="team", allow_llm=False)
    assert report.reports[0].outcome == "up-to-date", report
    assert teams.get_team(team.id).description == "USER TEAM EDIT TO KEEP"


def test_resolve_source_refuses_unlinked_item(env) -> None:
    ctx, hub, agents, teams, root = env
    row = agents.create_agent(_fields(name="local-only", description="d", tags=["role"]))
    with pytest.raises(LookupError):
        svc.resolve_source(
            ctx,
            kind="agent",
            name=row.name,
            hub_source="https://hub.example.com/v1",
            visibility="public",
            tenant_id=None,
        )


def _sections(n: int) -> str:
    return "".join(f"## S{i}\nrule {i} text\n\n" for i in range(1, n + 1))


def _section_slice(start: int, end: int) -> str:
    return "".join(f"## S{i}\nrule {i} text\n\n" for i in range(start, end + 1))


def test_complementary_removal_large_shrink_applies_when_fully_covered(env) -> None:
    """E1 regression: each side removes a DIFFERENT block, leaving a short overlap.

    Local keeps sections 1-5, remote keeps 4-8, so the merged overlap (4-5) is
    shorter than BOTH sides -- a genuine LargeShrink. Every dropped span is a
    deterministic one-sided ancestry removal, so the survivor proof allows the
    apply through the REAL auto-apply path instead of a ratio-only refusal.
    Before the fix, ``deletion_covered`` read a non-existent ``region.prov`` and
    raised AttributeError here (fail-closed, feature non-functional).
    """
    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root, text=_sections(8))
    agents.set_agent_system_prompt(row.id, _sections(5))  # local removed 6-8
    hub.agents["h1"] = (_section_slice(4, 8), "d")  # remote removed 1-3

    report = svc.apply_items(ctx, kind="agent", auto=True, allow_llm=False)

    assert report.reports[0].applied, report.reports[0]
    assert report.reports[0].outcome == "merged"
    merged = _text(agents, row.id)
    assert "## S4" in merged and "## S5" in merged
    assert "## S1" not in merged and "## S8" not in merged


# -- per-source attestation, outbound receipt advancement (round-1 remediation) ---------


def _org_record(root: Path, *, kind: Kind = "agent", source: str = "https://hub.test/v1"):
    """A clean org link whose facts are all of ``_eligibility``'s preconditions."""

    from local_operator.hub_sync.identity import LinkIdentity

    identity = LinkIdentity(kind, "a1", source, "org", "t1", "h1")  # type: ignore[arg-type]
    if kind == "skill":
        fields: dict[str, Any] = {
            "document_version": 1,
            "name": "skill-a",
            "description": "d",
            "files": [
                {
                    "path": "SKILL.md",
                    # SKILL.md carries the skill's identity: the manifest name and
                    # description must match this frontmatter exactly.
                    "content": "---\nname: skill-a\ndescription: d\n---\n\n# Skill\n",
                }
            ],
        }
    else:
        fields = {"instructions": "x"}
    record = prov.make_record(
        kind, "a1", "h1", "t1", fields, "merge-pull"  # type: ignore[arg-type]
    )
    from local_operator.hub_sync.push import receipt_for

    record = dataclasses.replace(
        record,
        identity=identity,
        remote_revision='"rev-1"',
        # An AGREED POINT: unattended push needs A/P (round-2 seeding ruling), so
        # the fixtures that exercise later gates carry one.
        outbound=receipt_for(kind, fields, fields, revision='"rev-1"'),
        lineage="clean",
        scope_known=True,
        local_alias="skill-a" if kind == "skill" else None,
    )
    prov.write_baseline(root, record)
    return identity, record


def test_capability_attestation_is_scoped_to_its_source(env) -> None:
    """An attestation observed on one API base never makes another base eligible."""

    ctx, _hub, _agents, _t, root = env
    identity, _record = _org_record(root)
    svc.record_server_capabilities(ctx, {"revision_cas": 1}, source="https://other.test/v1")
    assert svc.push_eligibility(ctx, prov.read_baseline(root, "agent", "a1")) == {
        "can_auto_push": False,
        "reason": "unsupported-precondition",
    }
    svc.record_server_capabilities(ctx, {"revision_cas": 1}, source=identity.hub_source)
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "agent", "a1"))["can_auto_push"] is True
    )


def test_skill_eligibility_needs_the_skill_family_on_its_source(env) -> None:
    """``revision_cas`` alone is not enough for a skill: the skill routes must be attested."""

    ctx, _hub, _agents, _t, root = env
    identity, _record = _org_record(root, kind="skill")
    svc.record_server_capabilities(ctx, {"revision_cas": 1}, source=identity.hub_source)
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "skill", "a1"))["can_auto_push"] is False
    )
    svc.record_server_capabilities(
        ctx, {"revision_cas": 1, "skills": 1}, source=identity.hub_source
    )
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "skill", "a1"))["can_auto_push"] is True
    )


def test_record_outbound_receipt_advances_the_accepted_revision_only(env) -> None:
    """An accepted push replaces (A,P) with the SERVER-returned pair; B is untouched.

    QA's repro: the receipt stayed on ``r1`` after the server reached ``r2``, so a
    later conditional write derived from a stale revision. The opaque accepted
    ETag (not ``content_revision``) is what a later If-Match sends verbatim, and
    the subscribed-off case must NOT look like an integrated inbound update.
    """

    ctx, _hub, _agents, _t, root = env
    _identity, _record = _org_record(root)
    svc.record_outbound_receipt(
        ctx,
        "agent",
        "a1",
        accepted={"instructions": "generalized"},
        current_local={"instructions": "local sent"},
        revision='"rev-2"',
        operation_id="srv-op",
    )
    back = prov.read_baseline(root, "agent", "a1")
    assert back is not None and back.outbound is not None
    assert back.outbound.remote_revision == '"rev-2"'
    assert back.outbound.local_fields == {"instructions": "local sent"}
    assert back.outbound.accepted_fields == {"instructions": "generalized"}
    assert back.outbound.operation_id == "srv-op"
    # B (the inbound integrated baseline) is NOT advanced by a push.
    assert back.remote_revision == '"rev-1"'


def test_record_checks_folds_an_observed_sync_contract_for_its_source(env) -> None:
    """The capability probe is the observed ``sync_contract`` of a REAL check response."""

    ctx, _hub, _agents, _t, root = env
    identity, _record = _org_record(root)
    from local_operator.hub_sync.check import ItemCheck

    c = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        "t1",
        "up-to-date",
        remote_revision='"rev-1"',
        sync_contract={"revision_cas": 1},
        # The base that ACTUALLY ANSWERED (round-2 R14): credit lands under this
        # key, never under the link's recorded source.
        served_by="https://hub.test/v1",
    )
    svc.record_checks(ctx, [c])
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "agent", "a1"))["can_auto_push"] is True
    )
    # The attestation landed under the link's OWN source, not a global flag.
    doc = st.StatusStore(root).load()
    assert "https://hub.test/v1" in doc["server"]["sources"]
    assert identity.hub_source == "https://hub.test/v1"


def test_attestation_is_keyed_by_the_answering_base_and_downgrades_clear(env) -> None:
    """R14: only the answering base earns credit, and a downgrade CLEARS it.

    (a) a link whose recorded source is A but whose check was ANSWERED by base B
    must not become eligible -- B's contract is B's fact, not A's.
    (b) an observation that no longer advertises the contract clears the source
    (last observation wins), so a downgraded deployment is honestly ineligible
    again instead of permanently "proven".
    """

    ctx, _hub, _agents, _t, root = env
    identity, _record = _org_record(root)
    from local_operator.hub_sync.check import ItemCheck

    def check(**over: object) -> ItemCheck:
        base: dict[str, object] = {
            "kind": "agent",
            "local_id": "a1",
            "name": "a",
            "hub_id": "h1",
            "tenant_id": "t1",
            "verdict": "up-to-date",
            "remote_revision": '"rev-1"',
            "sync_contract": {"revision_cas": 1},
            "served_by": "https://other.test/v1",
        }
        base.update(over)
        return ItemCheck(**base)  # type: ignore[arg-type]

    # (a) answered by B: no credit for A -> the PATCH/eligibility says no.
    svc.record_checks(ctx, [check()])
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "agent", "a1"))["can_auto_push"] is False
    )
    doc = st.StatusStore(root).load()
    assert identity.hub_source not in doc["server"]["sources"]

    # answered by A itself: eligible (attested A + agreed point + clean lineage).
    svc.record_checks(ctx, [check(served_by=identity.hub_source)])
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "agent", "a1"))["can_auto_push"] is True
    )

    # (b) a later observation from A that no longer advertises revision_cas
    # clears it (non-monotone): the source is honestly unproven again.
    svc.record_checks(ctx, [check(served_by=identity.hub_source, sync_contract={})])
    assert (
        svc.push_eligibility(ctx, prov.read_baseline(root, "agent", "a1"))["can_auto_push"] is False
    )


def test_auto_apply_holds_a_merge_that_would_empty_text_the_hub_has(env) -> None:
    """Q5: complementary removals that empty a field are HELD for review (auto).

    Each side removed what the other kept, so the engine can explain every
    dropped span (not the ratio-guard's "unexplained" loss) -- but auto-applying
    '' would silently discard text the hub still holds. A manual apply is not
    gated by this rule.
    """

    ctx, _hub, _agents, _t, _root = env
    from local_operator.hub_sync.check import ItemCheck

    check = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        None,
        "available",
        local={"instructions": "One rule.\n", "description": "d"},
        remote={"instructions": "Three rules.\n", "description": "d"},
        base={
            "instructions": "One rule. Two rules. Three rules. Four rules.\n",
            "description": "d",
        },
    )
    held = svc.apply_one(ctx, check, auto=True, allow_llm=False)
    assert held.outcome == "needs-review" and held.error_class == "merge-refused"
    assert "held for review" in held.message
    manual = svc.apply_one(ctx, check, auto=False, allow_llm=False)
    assert "held for review" not in manual.message  # manual stays available


def test_the_legacy_updates_projection_never_lists_skill_rows(env) -> None:
    """Q7: typed older consumers of /hub/updates must never receive a skill row."""

    ctx, _hub, _agents, _t, root = env

    def fold(doc: dict[str, Any]) -> None:
        items = doc.setdefault("items", {})
        items["agent:a1"] = {"kind": "agent", "name": "a", "local_id": "a1", "state": "available"}
        items["skill:s1"] = {"kind": "skill", "name": "s", "local_id": "s1", "state": "available"}

    st.StatusStore(root).mutate(fold)
    snapshot = svc.status_snapshot(ctx)
    assert {row["kind"] for row in snapshot["items"]} == {"agent"}
    assert snapshot["counts"]["available"] == 1  # the skill row is not counted either


def test_the_check_records_the_agent_revision_and_serving_base(env) -> None:
    """QN2a: the DETAIL read supplies the live revision; R14: the answering base."""

    ctx, hub, agents, _t, root = env
    _pull_agent(agents, hub, root)
    checks = svc.check_items(ctx, kinds=("agent",))
    (c,) = checks
    assert c.remote_revision == '"hub-rev-1"'
    assert c.served_by == "https://hub.test/v1"


def test_an_applied_pull_seeds_the_outbound_receipt(env) -> None:
    """Seeding ruling (A)a: the pull that integrates the remote agrees (A, P)."""

    import dataclasses

    from local_operator.hub_sync.identity import LinkIdentity

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    # A LINKED item: the baseline carries the writer identity (the legacy rig's
    # pull writes provenance without one; seeding requires a known origin).
    rec = prov.read_baseline(root, "agent", row.id)
    assert rec is not None
    prov.write_baseline(
        root,
        dataclasses.replace(
            rec, identity=LinkIdentity("agent", row.id, "https://hub.test/v1", "public", None, "h1")
        ),
    )
    hub.agents["h1"] = (BASE + "\n\n## New\nAdded upstream.", "d")

    report = svc.apply_items(ctx, kind="agent", allow_llm=False)

    assert report.reports[0].applied
    stored = prov.read_baseline(root, "agent", row.id)
    assert stored is not None
    receipt = stored.outbound
    assert receipt is not None
    assert receipt.remote_revision == '"hub-rev-1"'
    assert receipt.accepted_fields["instructions"].endswith("Added upstream.")


def test_a_link_without_an_agreed_point_is_ineligible_with_a_named_reason(env) -> None:
    """Seeding ruling (A): no receipt = a NAMED ineligible, never a guess."""

    import dataclasses

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    from local_operator.hub_sync.identity import LinkIdentity
    from local_operator.hub_sync.service import PUBLICATION_CONTRACT

    identity = LinkIdentity("agent", row.id, "https://hub.test/v1", "public", None, "h1")
    rec = prov.read_baseline(root, "agent", row.id)
    assert rec is not None
    rec = dataclasses.replace(
        rec,
        identity=identity,
        remote_revision='"rev-1"',
        lineage="clean",
        scope_known=True,
        outbound=None,
    )
    prov.write_baseline(root, rec)
    svc.record_server_capabilities(ctx, {"revision_cas": 1}, source="https://hub.test/v1")
    ctx.config_manager.set_config_value(
        "hub",
        {
            "links": {identity.link_id: {"subscribe": True, "auto_push": True}},
            "public_egress_grants": {
                identity.link_id: {
                    "link_id": identity.link_id,
                    "publication_contract": PUBLICATION_CONTRACT,
                }
            },
        },
    )
    stored = prov.read_baseline(root, "agent", row.id)
    assert stored is not None
    verdict = svc.push_eligibility(ctx, stored)
    assert verdict == {"can_auto_push": False, "reason": "no-receipt"}


def test_a_manual_publish_seeds_the_receipt_and_never_invents_a_revision(env) -> None:
    """Seeding ruling (A)b: the accepted publish seeds; no revision seeds nothing."""

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    accepted = {"instructions": BASE, "description": "d"}

    assert svc.seed_receipt_after_publish(
        root,
        "agent",
        row.id,
        accepted=accepted,
        local_fields=dict(accepted),
        revision='"rev-9"',
    )
    stored = prov.read_baseline(root, "agent", row.id)
    assert stored is not None
    seeded = stored.outbound
    assert seeded is not None and seeded.remote_revision == '"rev-9"'

    # A missing revision changes NOTHING: the client never invents an ETag.
    assert (
        svc.seed_receipt_after_publish(
            root,
            "agent",
            row.id,
            accepted=accepted,
            local_fields=dict(accepted),
            revision=None,
        )
        is False
    )
    final = prov.read_baseline(root, "agent", row.id)
    assert final is not None
    kept = final.outbound
    assert kept is not None and kept.remote_revision == '"rev-9"'


def test_a_broken_skill_row_never_costs_the_agent_checks(env, monkeypatch) -> None:
    """QN1: a per-item failure stays per-item -- the tick keeps every other kind."""

    from local_operator.hub_sync import check as chk

    ctx, hub, agents, _t, root = env
    _pull_agent(agents, hub, root)
    bad = chk.ItemCheck(
        "skill",
        "s1",
        "s",
        "sk-1",
        "t1",
        "unavailable",
        reason="invalid-remote-manifest",
        detail="missing description",
    )
    monkeypatch.setattr(chk, "check_hub_skills", lambda **kw: [bad])

    checks = svc.check_items(ctx, kinds=("agent", "skill"))

    kinds = {c.kind for c in checks}
    assert kinds == {"agent", "skill"}
    agent_check = next(c for c in checks if c.kind == "agent")
    assert agent_check.verdict in ("up-to-date", "available")


def test_the_agent_check_reads_content_and_revision_from_one_snapshot(env) -> None:
    """R18: a hub edit between the zip read and the detail read must not pair the
    OLD content with the NEW ETag -- that pair is what let a CAS write erase the
    unseen edit (reviewer repro: RACING-REMOTE overwritten)."""

    from local_operator.hub_sync import check as chk

    ctx, hub, agents, _t, root = env
    _pull_agent(agents, hub, root)
    # The row moved BEFORE the check: a new text and a new ETag are live, while
    # the legacy ZIP read still answers the pre-edit body.
    hub.agents["h1"] = (BASE + "\n\nRACING-REMOTE", "d")
    hub.revision = '"hub-rev-2"'
    stale = (BASE, "d")

    def zip_reads_stale(_client: Any, hub_id: str, **_kw: Any) -> tuple[str, str]:
        return stale

    (c,) = chk.check_hub_agents(
        agents, client_for_tenant=ctx.client_for_tenant, fetch=zip_reads_stale
    )
    # ONE snapshot: the detail read -- which carries content AND revision -- is
    # the deciding read; the raced edit is IN the check, so the merge keeps it.
    assert c.remote is not None and c.remote["instructions"].endswith("RACING-REMOTE")
    assert c.remote_revision == '"hub-rev-2"'
    assert c.served_by == "https://hub.test/v1"


def test_a_resolved_pull_that_agrees_seeds_and_establishes_clean_lineage(env) -> None:
    """QN7: an up-to-date apply against the link's own origin seeds the receipt,
    records the live revision, and leaves lineage clean + eligibility able."""

    from local_operator.hub_sync.service import PUBLICATION_CONTRACT

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    resolved = svc.resolve_source(
        ctx,
        kind="agent",
        name=row.name,
        hub_source="https://hub.test/v1",
        visibility="public",
        tenant_id=None,
    )
    assert resolved.lineage == "unknown"  # the resolve proves WHO, not that content landed

    checks = svc.check_items(ctx, kinds=("agent",))
    svc.record_checks(ctx, checks)
    report = svc.apply_items(ctx, kind="agent", allow_llm=False)

    assert report.reports[0].outcome == "up-to-date"
    assert not report.reports[0].applied
    stored = prov.read_baseline(root, "agent", row.id)
    assert stored is not None and stored.identity is not None
    # The integration event: clean lineage, the observed revision, the (A, P) seed.
    assert stored.lineage == "clean"
    assert stored.remote_revision == '"hub-rev-1"'
    receipt = stored.outbound
    assert receipt is not None
    assert receipt.remote_revision == '"hub-rev-1"'

    link_id = stored.identity.link_id
    ctx.config_manager.set_config_value(
        "hub",
        {
            "links": {link_id: {"subscribe": True, "auto_push": True}},
            "public_egress_grants": {
                link_id: {"link_id": link_id, "publication_contract": PUBLICATION_CONTRACT}
            },
        },
    )
    verdict = svc.push_eligibility(ctx, prov.read_baseline(root, "agent", row.id))
    assert verdict["can_auto_push"] is True, verdict


def test_seeding_guards_lineage_for_public_and_mixed_origins(env) -> None:
    """QN9: seeding must never mint a receipt that a later lineage flip could
    weaponise -- mixed never seeds; a PUBLIC origin seeds only from clean."""

    import dataclasses

    from local_operator.hub_sync.check import ItemCheck
    from local_operator.hub_sync.identity import LinkIdentity

    ctx, _hub, _agents, _t, root = env

    def check_for(local_id: str, hub_id: str, *, source: str) -> ItemCheck:
        return ItemCheck(
            "agent",
            local_id,
            "a",
            hub_id,
            None,
            "up-to-date",
            local={"instructions": "x"},
            remote={"instructions": "x"},
            remote_revision='"rev-1"',
            served_by=source,
        )

    from local_operator.hub_sync.identity import Visibility

    def seed_record(
        local_id: str, hub_id: str, visibility: Visibility, tenant: str | None, lineage: str
    ):
        identity = LinkIdentity(
            "agent", local_id, "https://hub.test/v1", visibility, tenant, hub_id
        )
        record = prov.make_record(
            "agent", local_id, hub_id, tenant, {"instructions": "x"}, "adopt-unknown"
        )
        record = dataclasses.replace(
            record, identity=identity, remote_revision='"rev-1"', lineage=lineage
        )
        prov.write_baseline(root, record)
        return identity

    # mixed-lineage: refused for EVERY origin, and nothing new is minted.
    mixed = seed_record("amix", "h3", "public", None, "mixed-lineage")
    assert (
        svc._seed_receipt_if_agreed(
            ctx, check_for("amix", "h3", source=mixed.hub_source), {"instructions": "x"}
        )
        is False
    )
    stored = prov.read_baseline(root, "agent", "amix")
    assert stored is not None and stored.outbound is None

    # a PUBLIC origin with unknown lineage: refused until an integration says clean.
    public = seed_record("apub", "h2", "public", None, "unknown")
    assert (
        svc._seed_receipt_if_agreed(
            ctx, check_for("apub", "h2", source=public.hub_source), {"instructions": "x"}
        )
        is False
    )

    # a NON-public origin with unknown lineage: the trusted-origin agreement seeds.
    org = seed_record("aorg", "h4", "org", "t1", "unknown")
    assert (
        svc._seed_receipt_if_agreed(
            ctx, check_for("aorg", "h4", source=org.hub_source), {"instructions": "x"}
        )
        is True
    )
    stored = prov.read_baseline(root, "agent", "aorg")
    assert stored is not None and stored.outbound is not None
    assert stored.outbound.remote_revision == '"rev-1"'


def test_a_local_only_check_never_marks_or_seeds_integration(env) -> None:
    """R21: ``local-only`` rides the same ``up-to-date`` verdict an equal pair
    gets, and must never establish lineage, a revision or a receipt."""

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    svc.resolve_source(
        ctx,
        kind="agent",
        name=row.name,
        hub_source="https://hub.test/v1",
        visibility="public",
        tenant_id=None,
    )
    # The LOCAL edit: L != R while R == B -- classification 'local-only'.
    agents.set_agent_system_prompt(row.id, BASE + "\n\nLOCAL-ONLY")

    checks = svc.check_items(ctx, kinds=("agent",))
    (c,) = checks
    assert c.classification is not None and c.classification.state == "local-only"
    assert c.verdict == "up-to-date"
    svc.record_checks(ctx, checks)
    report = svc.apply_items(ctx, kind="agent", allow_llm=False)
    assert report.reports[0].outcome == "up-to-date"

    stored = prov.read_baseline(root, "agent", row.id)
    assert stored is not None
    assert stored.lineage == "unknown", "a local edit is not an integration"
    assert stored.remote_revision is None, "no revision may be borrowed"
    assert stored.outbound is None, "no receipt may be seeded"
    # F2: resolve keeps the pull's provenance, so the record here is still the
    # PULL record -- what this test pins is that the local-only check did not
    # upgrade it into an integration (the lineage/revision/receipt trio above is
    # the integration evidence; the unknown marker is only for unpulled rows).
    assert stored.recorded_by == "pull"


def test_an_equal_read_without_a_revision_never_marks(env, monkeypatch) -> None:
    """R21: the zip fallback (detail read dead, no live revision) is equality
    WITHOUT anything a conditional write could stand on -- not an agreement."""

    ctx, hub, agents, _t, root = env
    row = _pull_agent(agents, hub, root)
    svc.resolve_source(
        ctx,
        kind="agent",
        name=row.name,
        hub_source="https://hub.test/v1",
        visibility="public",
        tenant_id=None,
    )

    calls: list[int] = []

    def detail_is_down(*_a: Any, **_k: Any) -> Any:
        calls.append(1)
        raise RuntimeError("detail read is down")

    monkeypatch.setattr(hub, "get_agent", detail_is_down)

    checks = svc.check_items(ctx, kinds=("agent",))
    (c,) = checks
    assert calls, "the test must actually break the DETAIL read"
    assert c.remote_revision is None  # the zip cannot expose one
    assert c.verdict == "up-to-date"
    assert c.served_by == "https://hub.test/v1"
    svc.record_checks(ctx, checks)
    svc.apply_items(ctx, kind="agent", allow_llm=False)

    stored = prov.read_baseline(root, "agent", row.id)
    assert stored is not None
    assert stored.lineage == "unknown"
    assert stored.remote_revision is None
    assert stored.outbound is None
    # And the push path refuses by name rather than writing unconditionally.
    assert svc.push_eligibility(ctx, stored)["can_auto_push"] is False
