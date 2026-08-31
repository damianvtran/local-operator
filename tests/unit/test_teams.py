"""Team registry: roster, layered briefs, and name rules."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    Team,
    TeamEditFields,
    TeamMember,
    TeamRegistry,
    parse_members,
)


@pytest.fixture()
def registry(tmp_path: Path) -> TeamRegistry:
    return TeamRegistry(tmp_path)


def test_create_list_show_update_delete(registry: TeamRegistry) -> None:
    team = registry.create_team(
        TeamEditFields(
            name="feature-release",
            description="Ship a user-facing change",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="reviewer", count=2)],
            instructions="Architect designs; coder implements; reviewer signs off.",
            project="user-dashboard",
        )
    )
    assert team.name == "feature-release"
    assert team.manager == "manager"
    assert [m.role for m in team.members] == ["coder", "reviewer"]
    assert team.members[1].count == 2
    assert "user-dashboard" in team.project

    listed = registry.list_teams()
    assert [row.name for row in listed] == ["feature-release"]
    assert registry.get_team_by_name("Feature-Release") is team or registry.get_team_by_name(
        "feature-release"
    )

    updated = registry.update_team(team.id, TeamEditFields(project="admin-api", instructions=None))
    assert updated.project == "admin-api"
    assert "Architect designs" in updated.instructions

    registry.delete_team(team.id)
    assert registry.list_teams() == []


def test_duplicate_name_is_refused(registry: TeamRegistry) -> None:
    registry.create_team(TeamEditFields(name="alpha", manager="manager"))
    with pytest.raises(ValueError, match="already exists"):
        registry.create_team(TeamEditFields(name="alpha", manager="manager"))


def test_name_rejects_spaces(registry: TeamRegistry) -> None:
    with pytest.raises(ValueError, match="team name"):
        registry.create_team(TeamEditFields(name="Feature Release", manager="manager"))


def test_parse_members_collapses_counts() -> None:
    slots = parse_members(["coder", "reviewer:2", "coder"])
    assert [(m.role, m.count) for m in slots] == [("coder", 2), ("reviewer", 2)]
    # A bare token stays an agent — the existing grammar is untouched.
    assert all(m.kind == "agent" for m in slots)


def test_parse_members_team_prefix() -> None:
    """``team:`` marks a nested-team slot; ``team:name:count`` carries a count."""
    slots = parse_members(["coder", "team:pod", "team:pod:2", "reviewer:2"])
    # ``team:pod`` and ``team:pod:2`` collapse into one team slot (summed).
    assert [(m.kind, m.role, m.count) for m in slots] == [
        ("agent", "coder", 1),
        ("team", "pod", 3),
        ("agent", "reviewer", 2),
    ]


def test_parse_members_team_prefix_is_case_insensitive() -> None:
    slots = parse_members(["TEAM:pod", "Team:qa:2"])
    assert [(m.kind, m.role, m.count) for m in slots] == [
        ("team", "pod", 1),
        ("team", "qa", 2),
    ]


def test_parse_members_agent_and_team_of_same_name_do_not_collapse() -> None:
    """A member ``pod`` and a sub-team ``pod`` are distinct slots, not merged."""
    slots = parse_members(["pod", "team:pod"])
    assert [(m.kind, m.role, m.count) for m in slots] == [
        ("agent", "pod", 1),
        ("team", "pod", 1),
    ]


def test_parse_members_team_prefix_without_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="no team name"):
        parse_members(["team:"])


def test_roster_lines_badges_team_slots() -> None:
    team = Team(
        id="x",
        name="org",
        created_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        manager="director",
        members=[
            TeamMember(role="coder"),
            TeamMember(role="pod", kind="team", count=2),
        ],
    )
    lines = team.roster_lines()
    assert lines[0] == "- manager: director (you, when this team is invoked)"
    assert "- coder" in lines[1]
    # A nested-team slot is badged so an org is distinguishable from a roster.
    assert lines[2] == "- pod (team) x2"


def test_fieldless_team_yml_loads_as_agent_kind(tmp_path: Path) -> None:
    """A pre-existing ``team.yml`` with no ``kind`` key loads to kind='agent'.

    This is the one backward-compat guarantee the model change rests on: an
    old file that predates nesting must validate unchanged, or ``_load``'s
    except-and-skip would silently drop the team. Written as a real on-disk
    file, not a model construction, because it is the DISK format that must
    survive the schema addition.
    """

    team_id = "legacy-id"
    team_dir = tmp_path / "teams" / team_id
    team_dir.mkdir(parents=True)
    (team_dir / "team.yml").write_text(
        "id: legacy-id\n"
        "name: legacy\n"
        "created_date: '2024-01-01T00:00:00+00:00'\n"
        "manager: manager\n"
        "members:\n"
        "- role: coder\n"
        "  count: 1\n"
        "- role: reviewer\n"
        "  count: 2\n",
        encoding="utf-8",
    )
    (team_dir / "instructions.md").write_text("", encoding="utf-8")
    (team_dir / "project.md").write_text("", encoding="utf-8")
    registry = TeamRegistry(tmp_path)
    loaded = registry.get_team_by_name("legacy")
    assert loaded is not None
    assert [m.role for m in loaded.members] == ["coder", "reviewer"]
    # The absent field defaults to "agent" — the roster means exactly what it
    # meant before nesting existed.
    assert all(m.kind == "agent" for m in loaded.members)


def test_manager_preamble_layers_briefs(registry: TeamRegistry) -> None:
    team = registry.create_team(
        TeamEditFields(
            name="release",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="Do not merge without review.",
            project="Keep the dashboard shipping weekly.",
        )
    )
    brief = team.manager_preamble()
    assert "[team: release]" in brief
    assert "coder" in brief
    assert "Do not merge without review." in brief
    assert "Keep the dashboard shipping weekly." in brief
    member = team.member_preamble("coder")
    assert "You are coder on this team" in member
    assert "Do not merge without review." in member


def test_oversized_briefs_are_refused(registry: TeamRegistry) -> None:
    with pytest.raises(ValueError, match="exceed"):
        registry.create_team(
            TeamEditFields(
                name="huge",
                manager="manager",
                instructions="x" * (MAX_TEAM_INSTRUCTIONS_CHARS + 1),
            )
        )


def test_reload_from_disk(tmp_path: Path) -> None:
    first = TeamRegistry(tmp_path)
    first.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            members=[TeamMember(role="scout")],
            project="on-call",
        )
    )
    second = TeamRegistry(tmp_path)
    loaded = second.get_team_by_name("ops")
    assert loaded is not None
    assert loaded.project == "on-call"
    assert loaded.members[0].role == "scout"


def test_list_is_metadata_only_and_get_loads_briefs_once(tmp_path: Path, monkeypatch) -> None:
    """Picker refreshes must not pay for either 8k brief until a full lookup."""
    first = TeamRegistry(tmp_path)
    team = first.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            instructions="Review before merging.",
            project="on-call",
        )
    )

    brief_paths = {
        tmp_path / "teams" / team.id / "instructions.md",
        tmp_path / "teams" / team.id / "project.md",
    }
    real_read_text = Path.read_text
    reads: list[Path] = []

    def tracked_read_text(path: Path, *args, **kwargs) -> str:
        if path in brief_paths:
            reads.append(path)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    second = TeamRegistry(tmp_path)
    listed = second.list_teams()
    assert [row.name for row in listed] == ["ops"]
    assert listed[0].instructions == ""
    assert listed[0].project == ""
    assert reads == []

    loaded = second.get_team_by_name("OPS")
    assert loaded is not None
    assert loaded.instructions == "Review before merging."
    assert loaded.project == "on-call"
    assert reads == [
        tmp_path / "teams" / team.id / "instructions.md",
        tmp_path / "teams" / team.id / "project.md",
    ]

    assert second.get_team(team.id) is loaded
    assert len(reads) == 2


def test_saving_a_metadata_only_list_result_preserves_both_briefs(tmp_path: Path) -> None:
    """R1-1 regression: a metadata-only ``list_teams`` row must not blank briefs.

    The exact reported repro: create a team with real briefs, reconstruct the
    registry (so ``list_teams`` hands back a team whose briefs were never
    read), mutate only its METADATA, save, reconstruct again — and both brief
    files must still hold their original text. Before the sentinel, the
    unloaded briefs read as ordinary ``""`` and the save wrote them back as
    empty files, destroying up to 16k of authored instructions in one
    metadata edit.
    """
    first = TeamRegistry(tmp_path)
    first.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            instructions="KEEP INSTRUCTIONS",
            project="KEEP PROJECT",
        )
    )

    second = TeamRegistry(tmp_path)
    listed = second.list_teams()[0]
    # The precondition the fix exists for: this object has NOT read the briefs.
    # Asserted through the registry, not the private class, so the test states
    # the contract ("a listed team is metadata-only") rather than the mechanism.
    assert listed.instructions == "" and listed.project == ""
    listed.description = "mutated, not the briefs"
    second.save_team(listed)

    third = TeamRegistry(tmp_path)
    reloaded = third.get_team_by_name("ops")
    assert reloaded is not None
    assert reloaded.instructions == "KEEP INSTRUCTIONS"
    assert reloaded.project == "KEEP PROJECT"
    assert reloaded.description == "mutated, not the briefs"
    # And the saved object itself no longer carries the unloaded state, so a
    # caller reading the briefs off its own save result sees the real text.
    assert listed.instructions == "KEEP INSTRUCTIONS"
    assert listed.project == "KEEP PROJECT"


def test_intentional_empty_brief_update_still_clears_the_files(tmp_path: Path) -> None:
    """The sentinel must not make an explicit clear impossible (R1-1's edge).

    ``update_team(..., TeamEditFields(instructions="", project=""))`` is the
    supported way to retire both briefs; pinning the preserve behaviour above
    without pinning this would let a future refactor trade one data-loss bug
    for an un-writable state.
    """
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            instructions="retire me",
            project="and me",
        )
    )
    registry.update_team(team.id, TeamEditFields(instructions="", project=""))

    reloaded = TeamRegistry(tmp_path)
    after = reloaded.get_team_by_name("ops")
    assert after is not None
    assert after.instructions == ""
    assert after.project == ""


def test_a_metadata_only_team_serializes_as_plain_empty_briefs() -> None:
    """No marker of any kind rides on the public brief fields.

    R2-1 retired the str-subclass sentinel: loaded/unloaded is registry
    state, so a ``Team`` — fresh, listed, or round-tripped — always carries
    ordinary strings. This pins the serialization contract the review
    exercised (``model_dump``/``model_dump_json``/``repr`` carry no marker)
    without depending on a sentinel existing to leak.
    """
    team = Team(
        id="t",
        name="ops",
        created_date=datetime.now(timezone.utc),
    )
    dumped = team.model_dump(mode="json")
    assert dumped["instructions"] == ""
    assert dumped["project"] == ""
    assert "unloaded" not in team.model_dump_json().lower()
    assert "unloaded" not in repr(team).lower()
    assert type(team.instructions) is str and type(team.project) is str


def _seed_keep_team(tmp_path: Path) -> str:
    """Create a team with both briefs set; returns its id."""
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(
            name="ops",
            manager="manager",
            instructions="KEEP INSTRUCTIONS",
            project="KEEP PROJECT",
        )
    )
    return team.id


def test_roundtripped_metadata_only_team_preserves_briefs_through_save(
    tmp_path: Path,
) -> None:
    """R2-1 regression 1: the Pydantic Python round trip must not truncate.

    The exact reviewed repro: a metadata-only ``list_teams`` row dumped with
    ``model_dump`` and revalidated with ``model_validate`` loses any private
    loaded/unloaded marker by construction — the registry owns that state —
    so ``save_team`` must still preserve both on-disk briefs when the
    round-tripped object carries only a METADATA edit.
    """
    _seed_keep_team(tmp_path)

    second = TeamRegistry(tmp_path)
    listed = second.list_teams()[0]
    transported = Team.model_validate(listed.model_dump())
    transported.description = "mutated, not the briefs"
    second.save_team(transported)

    third = TeamRegistry(tmp_path)
    reloaded = third.get_team_by_name("ops")
    assert reloaded is not None
    assert reloaded.instructions == "KEEP INSTRUCTIONS"
    assert reloaded.project == "KEEP PROJECT"
    assert reloaded.description == "mutated, not the briefs"


def test_roundtripped_metadata_only_team_preserves_briefs_through_json_save(
    tmp_path: Path,
) -> None:
    """R2-1 regression 2: the JSON round trip must not truncate either.

    ``model_dump_json``/``model_validate_json`` is the shape a tool or
    transport actually carries a team across; the reviewed blocker truncated
    both brief files through exactly this path.
    """
    _seed_keep_team(tmp_path)

    second = TeamRegistry(tmp_path)
    listed = second.list_teams()[0]
    transported = Team.model_validate_json(listed.model_dump_json())
    transported.manager = "lead"
    second.save_team(transported)

    third = TeamRegistry(tmp_path)
    reloaded = third.get_team_by_name("ops")
    assert reloaded is not None
    assert reloaded.instructions == "KEEP INSTRUCTIONS"
    assert reloaded.project == "KEEP PROJECT"
    assert reloaded.manager == "lead"


def test_directly_saved_metadata_list_object_preserves_briefs(tmp_path: Path) -> None:
    """R2-1 regression 3: no transport at all — the listed row itself.

    A caller can mutate the object ``list_teams`` handed it and pass it
    straight back; the registry must not treat its unread briefs as clears.
    """
    _seed_keep_team(tmp_path)

    second = TeamRegistry(tmp_path)
    listed = second.list_teams()[0]
    assert listed.instructions == "" and listed.project == ""
    listed.description = "metadata only edit"
    second.save_team(listed)

    third = TeamRegistry(tmp_path)
    reloaded = third.get_team_by_name("ops")
    assert reloaded is not None
    assert reloaded.instructions == "KEEP INSTRUCTIONS"
    assert reloaded.project == "KEEP PROJECT"
    # The saved object itself reads back the real text, not the "" it carried.
    assert listed.instructions == "KEEP INSTRUCTIONS"
    assert listed.project == "KEEP PROJECT"


def test_hydrated_team_brief_edits_persist_through_save(tmp_path: Path) -> None:
    """R2-1 regression 4: a LOADED team's deliberate brief edit survives.

    ``get_team*`` marks the id loaded, so its strings are authoritative: a
    caller that read the briefs, edited them, and saved must see its edit on
    disk — preservation must never swallow an authored change.
    """
    _seed_keep_team(tmp_path)

    second = TeamRegistry(tmp_path)
    team = second.get_team_by_name("ops")
    assert team is not None
    assert team.instructions == "KEEP INSTRUCTIONS"
    team.instructions = "REVISED COLLABORATION"
    team.project = "REVISED PROJECT"
    second.save_team(team)

    third = TeamRegistry(tmp_path)
    reloaded = third.get_team_by_name("ops")
    assert reloaded is not None
    assert reloaded.instructions == "REVISED COLLABORATION"
    assert reloaded.project == "REVISED PROJECT"


def test_update_team_explicit_empty_briefs_clear_the_files(tmp_path: Path) -> None:
    """R2-1 regression 5: the authored clear path still clears.

    ``update_team(..., TeamEditFields(instructions="", project=""))`` is the
    supported way to retire both briefs; it hydrates first, so the empty
    strings land on a loaded object and persist as empty files.
    """
    _seed_keep_team(tmp_path)

    registry = TeamRegistry(tmp_path)
    team = registry.get_team_by_name("ops")
    assert team is not None
    registry.update_team(team.id, TeamEditFields(instructions="", project=""))

    after = TeamRegistry(tmp_path).get_team_by_name("ops")
    assert after is not None
    assert after.instructions == ""
    assert after.project == ""


def test_update_name_and_briefs_share_one_immediately_refreshed_snapshot(tmp_path: Path) -> None:
    """R3-1: a rename cannot orphan the hydrated object before brief writes."""
    registry = TeamRegistry(tmp_path, refresh_interval=0)
    team = registry.create_team(TeamEditFields(name="ops", instructions="KEEP I", project="KEEP P"))

    updated = registry.update_team(
        team.id,
        TeamEditFields(name="renamed", instructions="", project="NEW P"),
    )
    assert (updated.name, updated.instructions, updated.project) == (
        "renamed",
        "",
        "NEW P",
    )

    reloaded = TeamRegistry(tmp_path).get_team_by_name("renamed")
    assert reloaded is not None
    assert (reloaded.name, reloaded.instructions, reloaded.project) == (
        "renamed",
        "",
        "NEW P",
    )


def test_update_name_collision_survives_immediately_due_refresh(tmp_path: Path) -> None:
    """R3-1: collision checks use the same refreshed snapshot as mutation."""
    registry = TeamRegistry(tmp_path, refresh_interval=0)
    first = registry.create_team(
        TeamEditFields(name="ops", instructions="KEEP I", project="KEEP P")
    )
    registry.create_team(TeamEditFields(name="other"))

    with pytest.raises(ValueError, match="Team with name other already exists"):
        registry.update_team(
            first.id,
            TeamEditFields(name="other", instructions="", project="NEW P"),
        )

    reloaded = TeamRegistry(tmp_path).get_team(first.id)
    assert (reloaded.name, reloaded.instructions, reloaded.project) == (
        "ops",
        "KEEP I",
        "KEEP P",
    )


def test_create_team_with_briefs_writes_them(tmp_path: Path) -> None:
    """R2-1 regression 6: the create path is not collateral damage.

    A brand-new id has no disk entry to preserve, so the caller-supplied
    briefs are the briefs: the fresh empty directory must never be read back
    over them.
    """
    registry = TeamRegistry(tmp_path)
    created = registry.create_team(
        TeamEditFields(
            name="fresh",
            manager="manager",
            instructions="NEW COLLABORATION",
            project="NEW PROJECT",
        )
    )
    assert created.instructions == "NEW COLLABORATION"
    assert created.project == "NEW PROJECT"

    reloaded = TeamRegistry(tmp_path).get_team_by_name("fresh")
    assert reloaded is not None
    assert reloaded.instructions == "NEW COLLABORATION"
    assert reloaded.project == "NEW PROJECT"


def test_a_fresh_registry_save_of_a_transport_roundtripped_loaded_team_preserves_disk(
    tmp_path: Path,
) -> None:
    """R2-1 edge: the transported object crosses registries.

    A model dumped by one registry and saved by a SECOND registry that never
    loaded the id still preserves the disk briefs: the destination registry
    has not read them, so it cannot treat the carried "" as authored.
    """
    _seed_keep_team(tmp_path)

    source = TeamRegistry(tmp_path)
    loaded = source.get_team_by_name("ops")
    assert loaded is not None
    transported = Team.model_validate_json(loaded.model_dump_json())
    # Simulate the transport dropping the loaded state: a second registry
    # receives only the model.
    destination = TeamRegistry(tmp_path)
    destination.save_team(transported)

    after = TeamRegistry(tmp_path).get_team_by_name("ops")
    assert after is not None
    assert after.instructions == "KEEP INSTRUCTIONS"
    assert after.project == "KEEP PROJECT"


def test_same_registry_does_not_trust_loadedness_for_a_transported_copy(tmp_path: Path) -> None:
    """R2-1 edge: loadedness belongs to the canonical object, not only its id.

    The registry may hydrate its canonical row, then receive a separately
    transported metadata-only model with the same id. That copy never carried
    loadedness, so its empty briefs must preserve disk rather than borrowing
    authority from the registry's earlier lookup.
    """
    _seed_keep_team(tmp_path)

    registry = TeamRegistry(tmp_path)
    metadata = registry.list_teams()[0]
    transported = Team.model_validate_json(metadata.model_dump_json())
    canonical = registry.get_team(metadata.id)
    assert canonical.instructions == "KEEP INSTRUCTIONS"
    transported.description = "safe metadata edit"
    registry.save_team(transported)

    after = TeamRegistry(tmp_path).get_team_by_name("ops")
    assert after is not None
    assert after.instructions == "KEEP INSTRUCTIONS"
    assert after.project == "KEEP PROJECT"
    assert after.description == "safe metadata edit"
