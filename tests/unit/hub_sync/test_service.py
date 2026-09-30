"""End-to-end merge/apply on REAL registries with a fake hub: the never-overwrite guarantees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.merge import ConflictProposal, ResolverError
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

BASE = "## Rules\nBe brief.\n\n## Legacy\nOld advice.\n\n## Tools\nUse tools carefully."


class Hub:
    """A fake hub serving one agent listing and any number of team documents."""

    def __init__(self) -> None:
        self.agents: dict[str, tuple[str, str]] = {}
        self.teams: dict[str, dict[str, Any]] = {}

    def client(self) -> Any:
        return self

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
