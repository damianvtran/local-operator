"""Team registry: roster, layered briefs, and name rules."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Thread

import pytest
import yaml

import local_operator.teams as teams_module
from local_operator.teams import (
    MAX_TEAM_INSTRUCTIONS_CHARS,
    Team,
    TeamEditFields,
    TeamMember,
    TeamRegistry,
    TeamRegistryLockTimeout,
    TeamRegistryRecoveryError,
    parse_members,
)


@pytest.fixture()
def registry(tmp_path: Path) -> TeamRegistry:
    return TeamRegistry(tmp_path)


def test_constructing_unused_registry_does_not_create_config_tree(tmp_path: Path) -> None:
    config_dir = tmp_path / "missing"
    TeamRegistry(config_dir)

    assert not config_dir.exists()


@pytest.mark.parametrize(
    "hostile_id",
    [
        "./../../victim",
        "../escaped",
        "folder/child",
        r"folder\\child",
        "/tmp/absolute",
        ".",
        "..",
    ],
)
def test_team_model_rejects_path_ids(hostile_id: str) -> None:
    with pytest.raises(ValueError, match="team id"):
        Team(
            id=hostile_id,
            name="hostile",
            created_date=datetime.now(timezone.utc),
        )


def test_save_defensively_rejects_bypassed_path_id_without_touching_disk(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "user-data"
    sentinel.write_text("keep", encoding="utf-8")
    registry = TeamRegistry(tmp_path / "config")
    hostile = Team.model_construct(
        id=str(victim),
        name="hostile",
        created_date=datetime.now(timezone.utc),
        description="",
        manager="manager",
        members=[],
        instructions="",
        project="",
    )

    with pytest.raises(ValueError, match="team id"):
        registry.save_team(hostile)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not registry.config_dir.exists()


def test_load_skips_symlinked_rows_and_metadata(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    row_id = "11111111-2222-3333-4444-555555555555"
    (outside / "team.yml").write_text(
        yaml.safe_dump(
            {
                "id": row_id,
                "name": "outside",
                "created_date": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    teams = tmp_path / "config" / "teams"
    teams.mkdir(parents=True)
    (teams / row_id).symlink_to(outside, target_is_directory=True)

    assert TeamRegistry(tmp_path / "config").list_teams() == []


def test_lock_sidecar_is_not_a_team_row(tmp_path: Path) -> None:
    registry = TeamRegistry(tmp_path)
    registry.create_team(TeamEditFields(name="ops"))

    assert (tmp_path / ".teams.lock").is_file()
    assert sorted(path.name for path in (tmp_path / "teams").iterdir()) == [
        registry.list_teams()[0].id
    ]


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


def test_case_insensitive_name_collision_is_refused(registry: TeamRegistry) -> None:
    registry.create_team(TeamEditFields(name="Beta", manager="manager"))
    with pytest.raises(ValueError, match="Team with name beta already exists"):
        registry.create_team(TeamEditFields(name="beta", manager="manager"))


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
    # R6-2 pins all three files before reading so backup cleanup cannot unlink
    # one between sequential opens. Track that seam rather than Path.read_text.
    real_open_at = teams_module._open_optional_at
    fd_reads: list[str] = []

    def tracked_open_at(directory_fd: int, filename: str) -> int | None:
        fd_reads.append(filename)
        return real_open_at(directory_fd, filename)

    monkeypatch.setattr(teams_module, "_open_optional_at", tracked_open_at)
    second = TeamRegistry(tmp_path)
    listed = second.list_teams()
    assert [row.name for row in listed] == ["ops"]
    assert listed[0].instructions == ""
    assert listed[0].project == ""
    assert reads == []
    assert fd_reads == []

    loaded = second.get_team_by_name("OPS")
    assert loaded is not None
    assert loaded.instructions == "Review before merging."
    assert loaded.project == "on-call"
    assert sorted(fd_reads) == ["instructions.md", "project.md", "team.yml", "team.yml"]

    assert second.get_team(team.id) is loaded
    assert len(fd_reads) == 4


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


def test_direct_json_save_refuses_rename_to_an_occupied_name(tmp_path: Path) -> None:
    """R4-1: transported saves enforce uniqueness at the disk boundary."""
    registry = TeamRegistry(tmp_path)
    alpha = registry.create_team(TeamEditFields(name="alpha", instructions="ALPHA"))
    beta = registry.create_team(TeamEditFields(name="beta", instructions="BETA"))
    transported = Team.model_validate_json(alpha.model_dump_json())
    transported.name = "beta"

    with pytest.raises(ValueError, match="Team with name beta already exists"):
        registry.save_team(transported)

    reloaded = TeamRegistry(tmp_path)
    assert {(team.id, team.name) for team in reloaded.list_teams()} == {
        (alpha.id, "alpha"),
        (beta.id, "beta"),
    }
    assert reloaded.get_team(alpha.id).instructions == "ALPHA"
    assert reloaded.get_team(beta.id).instructions == "BETA"


def test_stale_deleted_id_save_cannot_recreate_an_occupied_name(tmp_path: Path) -> None:
    """R4-1: delete/recreate wins over a stale transported old-id save."""
    registry = TeamRegistry(tmp_path)
    old = registry.create_team(TeamEditFields(name="ops", project="OLD"))
    stale = Team.model_validate_json(old.model_dump_json())
    registry.delete_team(old.id)
    new = registry.create_team(TeamEditFields(name="ops", project="NEW"))

    with pytest.raises(ValueError, match="Team with name ops already exists"):
        registry.save_team(stale)

    rows = TeamRegistry(tmp_path).list_teams()
    assert [(team.id, team.name) for team in rows] == [(new.id, "ops")]
    assert not (tmp_path / "teams" / old.id).exists()


def test_concurrent_registries_serialize_same_name_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-1: synchronized stale snapshots still publish exactly one row.

    The wrapped refresh is the pre-fix race seam: without persistence locking,
    both threads adopt the empty snapshot before either checks or writes. With
    the lock, the first bounded wait expires and publishes; the second refreshes
    afterward and sees that row. Catching ``BrokenBarrierError`` is deliberate
    because lock serialization means both threads cannot reach the seam together.
    """
    first = TeamRegistry(tmp_path)
    second = TeamRegistry(tmp_path)
    barrier = Barrier(2)

    # Pre-fix create refreshed here and then checked this snapshot outside any
    # lock. Synchronizing the two refreshes makes both callers hold the same
    # empty view before either can continue to collision-check and save.
    for registry in (first, second):
        real_load = registry._load

        def synchronized_load(*, _real_load=real_load):
            _real_load()
            try:
                barrier.wait(timeout=0.25)
            except BrokenBarrierError:
                pass

        monkeypatch.setattr(registry, "_load", synchronized_load)

    outcomes: list[Team | Exception] = []

    def create(registry: TeamRegistry) -> None:
        try:
            outcomes.append(registry.create_team(TeamEditFields(name="Same")))
        except Exception as exc:  # noqa: BLE001 - the assertion checks the public outcome
            outcomes.append(exc)

    threads = [Thread(target=create, args=(registry,)) for registry in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert sum(isinstance(outcome, Team) for outcome in outcomes) == 1
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "already exists" in str(errors[0])
    rows = TeamRegistry(tmp_path).list_teams()
    assert len(rows) == 1
    assert rows[0].name.casefold() == "same"


def test_same_id_update_persists_under_registry_lock(tmp_path: Path) -> None:
    """R4-1: collision enforcement must still permit same-row updates."""
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="ops", instructions="OLD"))

    updated = registry.update_team(
        team.id,
        TeamEditFields(name="OPS", description="revised", instructions="NEW"),
    )

    reloaded = TeamRegistry(tmp_path).get_team(team.id)
    assert (updated.id, reloaded.name, reloaded.description, reloaded.instructions) == (
        team.id,
        "OPS",
        "revised",
        "NEW",
    )


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


# --- R5-1: staging invisibility and directory/id agreement ------------------


def test_reader_during_create_sees_no_row_until_publish(tmp_path: Path) -> None:
    """R5-1: a paused writer's staged team.yml is invisible to other registries.

    The writer is paused immediately after staging ``team.yml`` — before the
    briefs and before the publish rename. A concurrent reader must not see the
    row at all: pre-fix, ``_load`` trusted the staged YAML, hydrated "" briefs
    as canonical, and a later save of that object erased the authored briefs.
    """
    import local_operator.teams as module

    writer = TeamRegistry(tmp_path)
    paused = threading.Event()
    release = threading.Event()
    real_write = module._write_row_files

    def paused_write(directory, metadata, team):
        real_write(directory, metadata, team)
        if directory.parent == writer.teams_dir:
            paused.set()
            release.wait(timeout=10)

    module._write_row_files = paused_write
    try:
        thread = threading.Thread(
            target=lambda: writer.create_team(
                TeamEditFields(name="race", instructions="KEEP-I", project="KEEP-P")
            )
        )
        thread.start()
        assert paused.wait(timeout=5), "writer never staged the row"

        # The exact window the blocker found: staged team.yml exists on disk,
        # the row is not published. A concurrent registry must see NOTHING.
        staged = next(p for p in writer.teams_dir.iterdir() if p.name.startswith("."))
        assert (staged / "team.yml").is_file()
        reader = TeamRegistry(tmp_path)
        assert reader.list_teams() == []
        assert reader.get_team_by_name("race") is None

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        module._write_row_files = real_write

    fresh = TeamRegistry(tmp_path)
    seen = fresh.get_team_by_name("race")
    assert seen is not None
    assert seen.instructions == "KEEP-I"
    assert seen.project == "KEEP-P"


def test_saving_a_row_seen_during_staging_cannot_erase_authored_briefs(
    tmp_path: Path,
) -> None:
    """R5-1 belt-and-braces: even a poisoned cache cannot truncate the briefs.

    Pre-fix, a reader in the create gap hydrated a metadata-only row and its
    later save wrote "" over both authored briefs. Post-fix the row is not
    visible in the gap at all — this test goes further and injects the poisoned
    cache state directly, then saves it AFTER the creator finished. The save
    must preserve the authored briefs: hydration reads them from disk.
    """
    import local_operator.teams as module

    writer = TeamRegistry(tmp_path)
    paused = threading.Event()
    release = threading.Event()
    real_write = module._write_row_files

    def paused_write(directory, metadata, team):
        real_write(directory, metadata, team)
        if directory.parent == writer.teams_dir:
            paused.set()
            release.wait(timeout=10)

    module._write_row_files = paused_write
    poisoned: dict[str, object] = {}
    try:
        thread = threading.Thread(
            target=lambda: writer.create_team(
                TeamEditFields(name="race", instructions="KEEP-I", project="KEEP-P")
            )
        )
        thread.start()
        assert paused.wait(timeout=5)

        # The pre-fix reader's exact state, reconstructed by hand: a
        # metadata-only object adopted into the cache while the row was staged.
        staged_dir = next(p for p in writer.teams_dir.iterdir() if p.name.startswith("."))
        staged = Team.model_validate(yaml.safe_load((staged_dir / "team.yml").read_text()))
        reader = TeamRegistry(tmp_path)
        reader._teams[staged.id] = staged
        poisoned["id"] = staged.id

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        # The hostile save: metadata edit on the never-hydrated object.
        staged.description = "hostile metadata edit"
        reader.save_team(staged)
    finally:
        module._write_row_files = real_write

    fresh = TeamRegistry(tmp_path)
    seen = fresh.get_team(str(poisoned["id"]))
    assert seen.instructions == "KEEP-I"
    assert seen.project == "KEEP-P"
    assert seen.description == "hostile metadata edit"


def test_crash_staging_artifact_with_valid_team_yml_stays_hidden(tmp_path: Path) -> None:
    """R5-1: a crash-left staging directory is ignored even with valid metadata.

    Pre-fix, a staged (or orphaned) directory whose team.yml named a team made
    that team discoverable. Post-fix, any dot-prefixed sibling of ``teams/``
    is not a row — forever — because no rename ever published it.
    """
    teams = tmp_path / "teams"
    teams.mkdir()
    staging = teams / ".11111111-2222-3333-4444-555555555555.crashed"
    staging.mkdir()
    (staging / "team.yml").write_text(
        yaml.safe_dump(
            {
                "id": "11111111-2222-3333-4444-555555555555",
                "name": "ghost",
                "created_date": "2026-01-01T00:00:00Z",
                "manager": "manager",
                "members": [],
            }
        ),
        encoding="utf-8",
    )

    registry = TeamRegistry(tmp_path)
    assert registry.list_teams() == []
    assert registry.get_team_by_name("ghost") is None


def test_published_directory_name_must_match_metadata_id(tmp_path: Path) -> None:
    """R5-1: a directory whose YAML id points elsewhere is skipped as invalid."""
    teams = tmp_path / "teams"
    teams.mkdir()
    good = teams / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    good.mkdir()
    (good / "team.yml").write_text(
        yaml.safe_dump(
            {
                "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "name": "real",
                "created_date": "2026-01-01T00:00:00Z",
                "manager": "manager",
                "members": [],
            }
        ),
        encoding="utf-8",
    )
    # Directory name disagrees with the id inside its team.yml: the row would
    # load under a cache key whose briefs and deletion target another path.
    bad = teams / "ffffffff-0000-0000-0000-000000000000"
    bad.mkdir()
    (bad / "team.yml").write_text(
        yaml.safe_dump(
            {
                "id": "99999999-9999-9999-9999-999999999999",
                "name": "phantom",
                "created_date": "2026-01-01T00:00:00Z",
                "manager": "manager",
                "members": [],
            }
        ),
        encoding="utf-8",
    )

    registry = TeamRegistry(tmp_path)
    assert [team.name for team in registry.list_teams()] == ["real"]


# --- R5-2: directory-level transaction for existing rows ---------------------


def test_hydration_adopts_complete_metadata_and_briefs_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = TeamRegistry(tmp_path)
    team = writer.create_team(
        TeamEditFields(
            name="ops",
            manager="old-manager",
            members=[TeamMember(role="old-role")],
            instructions="old-I",
            project="old-P",
        )
    )
    reader = TeamRegistry(tmp_path)
    writer.update_team(
        team.id,
        TeamEditFields(
            manager="new-manager",
            members=[TeamMember(role="new-role", count=2)],
            instructions="new-I",
            project="new-P",
        ),
    )

    full = reader.get_team(team.id)
    assert (
        full.manager,
        [(member.role, member.count) for member in full.members],
        full.instructions,
        full.project,
    ) == ("new-manager", [("new-role", 2)], "new-I", "new-P")

    # Force the non-dirfd branch against another stale cache. Its bounded
    # metadata-before/briefs/metadata-after snapshot must adopt one whole row.
    fallback = TeamRegistry(tmp_path)
    writer.update_team(
        team.id,
        TeamEditFields(
            manager="final-manager",
            members=[TeamMember(role="final-role")],
            instructions="final-I",
            project="final-P",
        ),
    )
    monkeypatch.setattr(teams_module, "_DIR_FD_READS", False)
    full = fallback.get_team(team.id)
    assert (
        full.manager,
        [member.role for member in full.members],
        full.instructions,
        full.project,
    ) == ("final-manager", ["final-role"], "final-I", "final-P")


def test_fallback_hydration_retries_swap_during_brief_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(name="ops", manager="old", instructions="old-I", project="old-P")
    )
    reader = TeamRegistry(tmp_path)
    replacement = Team(
        id=team.id,
        name="ops",
        created_date=team.created_date,
        manager="new",
        instructions="new-I",
        project="new-P",
    )
    target = tmp_path / "teams" / team.id
    staging = tmp_path / "teams" / ".replacement"
    staging.mkdir()
    payload = replacement.model_dump(mode="json", exclude={"instructions", "project"})
    teams_module._write_row_files(staging, yaml.safe_dump(payload, sort_keys=False), replacement)

    real_read = teams_module._read_optional_strict
    swapped = False

    def swap_after_instructions(path: Path) -> str:
        nonlocal swapped
        value = real_read(path)
        if path.name == "instructions.md" and not swapped:
            swapped = True
            old = tmp_path / "teams" / ".old"
            os.replace(target, old)
            os.replace(staging, target)
        return value

    monkeypatch.setattr(teams_module, "_DIR_FD_READS", False)
    monkeypatch.setattr(teams_module, "_read_optional_strict", swap_after_instructions)
    full = reader.get_team(team.id)
    assert swapped
    assert (full.manager, full.instructions, full.project) == ("new", "new-I", "new-P")


@pytest.mark.parametrize(
    "failure_at",
    [
        "staged_metadata",
        "staged_instructions",
        "staged_project",
        "target_aside",
        "publish",
    ],
)
def test_injected_update_failure_preserves_old_complete_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    """R5-2: failure after each staged write and each swap step keeps the row whole.

    Old row is (old, old-desc, old-inst, old-proj); the update targets
    (new, new-desc, new-inst, new-proj). Whichever step is made to fail, a
    FRESH registry must read the OLD complete row and ``teams/`` must hold no
    dot-prefixed leftovers. ``os.replace`` is the seam for the swap steps;
    ``_write_row_files`` is the seam for the staged writes (its fsync-per-file
    design means a mid-row failure leaves the staging directory incomplete,
    which the caller's ``except`` removes before the error surfaces).
    """
    import local_operator.teams as module

    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(
            name="old",
            description="old-desc",
            instructions="old-inst",
            project="old-proj",
        )
    )
    staged_writes: list[str] = []
    swap_renames: list[str] = []

    real_write = module._write_row_files
    row_files = ("team.yml", "instructions.md", "project.md")

    def failing_write(directory, metadata, team):
        # Only writes into a STAGING directory (dot-prefixed sibling of
        # teams/) count; nothing else in the process is disturbed.
        if directory.parent != registry.teams_dir or not directory.name.startswith("."):
            real_write(directory, metadata, team)
            return
        stage_failures = {
            "staged_metadata": 0,
            "staged_instructions": 1,
            "staged_project": 2,
        }
        if failure_at not in stage_failures:
            real_write(directory, metadata, team)
            return
        target_index = stage_failures[failure_at]
        staged_writes.append(row_files[target_index])
        # Fail DURING the staged row: the named file and everything before it
        # are on disk, everything after is not — the exact mid-row crash.
        if target_index < 2:
            real_partial = module._write_row_files_after
            real_partial(directory, metadata, team, stop_after=row_files[target_index])
            raise OSError(f"injected failure after staged {row_files[target_index]}")
        real_write(directory, metadata, team)
        raise OSError("injected failure after staged project.md")

    real_replace = module.os.replace

    def failing_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        src_path, dst_path = Path(str(src)), Path(str(dst))
        in_teams = src_path.parent == registry.teams_dir
        staging_source = in_teams and src_path.name.startswith(".")
        live_source = in_teams and src_path.name == team.id
        backup_destination = dst_path.parent == registry.teams_dir and dst_path.name.startswith(".")
        if live_source and backup_destination and failure_at == "target_aside":
            swap_renames.append("aside")
            raise OSError("injected failure at target->backup")
        if (
            staging_source
            and not backup_destination
            and dst_path.name == team.id
            and failure_at == "publish"
        ):
            swap_renames.append("publish")
            raise OSError("injected failure at staged->target")
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(module, "_write_row_files", failing_write)
    monkeypatch.setattr(module.os, "replace", failing_replace)

    with pytest.raises(Exception, match="Failed to save team metadata"):
        registry.update_team(
            team.id,
            TeamEditFields(
                name="new",
                description="new-desc",
                instructions="new-inst",
                project="new-proj",
            ),
        )

    monkeypatch.undo()
    # The long-lived registry must remain on the acknowledged old row too.
    same = registry.get_team_by_name("old")
    assert same is not None
    assert (same.name, same.description, same.instructions, same.project) == (
        "old",
        "old-desc",
        "old-inst",
        "old-proj",
    )

    fresh = TeamRegistry(tmp_path)
    seen = fresh.get_team_by_name("old")
    assert seen is not None, "old row vanished entirely"
    assert (seen.name, seen.description, seen.instructions, seen.project) == (
        "old",
        "old-desc",
        "old-inst",
        "old-proj",
    )
    leftovers = [p.name for p in (tmp_path / "teams").iterdir() if p.name.startswith(".")]
    assert leftovers == []

    retried = registry.update_team(
        team.id,
        TeamEditFields(name="new", instructions="new-inst", project="new-proj"),
    )
    assert (retried.name, retried.instructions, retried.project) == (
        "new",
        "new-inst",
        "new-proj",
    )


@pytest.mark.parametrize("fail_call", [1, 2])
def test_update_directory_fsync_failure_rolls_back_cache_and_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_call: int
) -> None:
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(name="old", manager="old-lead", instructions="old-I", project="old-P")
    )
    real_fsync_dir = teams_module._fsync_dir
    calls = 0

    def fail_selected(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise OSError(errno.EIO, "durability failure")
        real_fsync_dir(path)

    monkeypatch.setattr(teams_module, "_fsync_dir", fail_selected)
    with pytest.raises(Exception, match="Failed to save team metadata"):
        registry.update_team(
            team.id,
            TeamEditFields(name="new", manager="new-lead", instructions="new-I", project="new-P"),
        )
    monkeypatch.undo()

    same = registry.get_team(team.id)
    fresh = TeamRegistry(tmp_path).get_team(team.id)
    for seen in (same, fresh):
        assert (seen.name, seen.manager, seen.instructions, seen.project) == (
            "old",
            "old-lead",
            "old-I",
            "old-P",
        )


def test_crash_between_swap_renames_recovers_on_next_locked_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5-2: target->backup done, staged->target never ran — recovery restores.

    The injected ``KeyboardInterrupt`` simulates process death at the exact
    gap: the live row exists only under its hidden backup name. The next
    locked mutation (here, an unrelated create) runs recovery first, so the
    stranded row is restored whole before anything else observes the registry.
    """
    import local_operator.teams as module

    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(name="old", instructions="old-inst", project="old-proj")
    )
    renames: list[str] = []
    real_replace = module.os.replace

    def die_at_publish(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        src_path, dst_path = Path(str(src)), Path(str(dst))
        in_teams = src_path.parent == registry.teams_dir
        staging_source = in_teams and src_path.name.startswith(".")
        backup_destination = dst_path.parent == registry.teams_dir and dst_path.name.startswith(".")
        if backup_destination:
            renames.append("aside")
            return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        if staging_source and dst_path.name == team.id and "publish" not in renames:
            renames.append("publish")
            raise KeyboardInterrupt  # process death between the two renames
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(module.os, "replace", die_at_publish)
    with pytest.raises(BaseException):
        registry.update_team(team.id, TeamEditFields(instructions="new-inst"))
    assert renames == ["aside", "publish"]

    # The writer CAUGHT the interrupt and rolled the backup back into place,
    # so a fresh reader sees the OLD complete row (never a mixed one). The
    # true process-death variant is covered below by simulating the stranded
    # state directly.
    monkeypatch.undo()
    survivor = TeamRegistry(tmp_path)
    seen = survivor.get_team(team.id)
    assert (seen.name, seen.instructions, seen.project) == ("old", "old-inst", "old-proj")
    assert [p.name for p in (tmp_path / "teams").iterdir() if p.name.startswith(".")] == []

    # True crash state: target gone, row only in the hidden backup. The first
    # fresh read now recovers it under lock instead of reporting a false absence.
    row_dir = tmp_path / "teams" / team.id
    backup_dir = tmp_path / "teams" / f".{team.id}.backup.stranded"
    row_dir.rename(backup_dir)
    recovered = TeamRegistry(tmp_path).list_teams()
    assert [row.name for row in recovered] == ["old"]

    healed = TeamRegistry(tmp_path)
    healed.create_team(TeamEditFields(name="other"))
    restored = healed.get_team(team.id)
    assert restored.instructions == "old-inst"
    assert restored.project == "old-proj"
    names = sorted(p.name for p in (tmp_path / "teams").iterdir())
    assert names and all(not name.startswith(".") for name in names)


def test_backup_left_beside_newer_target_is_cleaned_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5-2: a crash after publish leaves target authoritative, backup cleaned.

    The swap completed (new revision is live) but the backup removal did not.
    A later locked mutation must treat the target as authoritative and remove
    the stale hidden backup rather than restore the older revision over it.
    """
    import local_operator.teams as module

    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="row", instructions="I", project="P"))
    real_rmtree = module.shutil.rmtree

    def die_on_backup_cleanup(path, *args, **kwargs):
        if Path(str(path)).name.startswith(".") and ".backup." in Path(str(path)).name:
            raise KeyboardInterrupt  # death between publish and cleanup
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", die_on_backup_cleanup)
    with pytest.raises(BaseException):
        registry.update_team(team.id, TeamEditFields(instructions="I2"))
    monkeypatch.undo()

    # Both the new target and the stale backup exist right now.
    leftovers = [p.name for p in (tmp_path / "teams").iterdir() if ".backup." in p.name]
    assert len(leftovers) == 1

    healed = TeamRegistry(tmp_path)
    seen = healed.get_team(team.id)
    assert seen.instructions == "I2"
    healed.create_team(TeamEditFields(name="unrelated"))
    assert [p.name for p in (tmp_path / "teams").iterdir() if ".backup." in p.name] == []


def test_new_row_create_is_published_atomically(tmp_path: Path) -> None:
    """R5-2: the create path still publishes by one directory rename.

    A create must never leave a visible directory without its complete file
    set; the staging name is dot-prefixed and removed on any failure.
    """
    import local_operator.teams as module

    registry = TeamRegistry(tmp_path)
    real_replace = module.os.replace
    seen: list[str] = []

    def recording_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        src_path, dst_path = Path(str(src)), Path(str(dst))
        if dst_path.parent == registry.teams_dir and not dst_path.name.startswith("."):
            staged = src_path.parent == registry.teams_dir and src_path.name.startswith(".")
            seen.append(f"{staged}->{dst_path.name}")
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.os, "replace", recording_replace)
    try:
        team = registry.create_team(TeamEditFields(name="fresh", instructions="FI", project="FP"))
    finally:
        monkeypatch.undo()

    assert seen == [f"True->{team.id}"]
    row = tmp_path / "teams" / team.id
    assert sorted(p.name for p in row.iterdir()) == [
        "instructions.md",
        "project.md",
        "team.yml",
    ]


# --- U5-1: lock timeout surface ---------------------------------------------


@pytest.mark.parametrize("failure_errno", [errno.EIO, errno.ENOSPC])
def test_directory_fsync_propagates_real_storage_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_errno: int
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    monkeypatch.setattr(
        os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError(failure_errno, "fail"))
    )
    with pytest.raises(OSError) as excinfo:
        teams_module._fsync_dir(directory)
    assert excinfo.value.errno == failure_errno


def test_directory_fsync_suppresses_unsupported_einval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    monkeypatch.setattr(
        os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError(errno.EINVAL, "unsupported"))
    )
    teams_module._fsync_dir(directory)


def test_create_fsync_failure_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = TeamRegistry(tmp_path)
    real_fsync_dir = teams_module._fsync_dir
    calls = 0

    def fail_first_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOSPC, "full")
        real_fsync_dir(path)

    monkeypatch.setattr(teams_module, "_fsync_dir", fail_first_fsync)
    with pytest.raises(Exception, match="Failed to save team metadata"):
        registry.create_team(TeamEditFields(name="not-published"))
    assert registry.list_teams() == []
    assert TeamRegistry(tmp_path).list_teams() == []


def _strand_backup(registry: TeamRegistry, team_id: str) -> Path:
    target = registry.teams_dir / team_id
    backup = registry.teams_dir / f".{team_id}.backup.interrupted"
    target.rename(backup)
    return backup


def _run_teams_cli(config_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            "teams",
            *args,
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def test_fresh_read_recovers_stranded_backup_complete_row(tmp_path: Path) -> None:
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(
        TeamEditFields(
            name="survivor",
            description="durable",
            manager="lead",
            members=[TeamMember(role="coder", count=2)],
            instructions="collaborate",
            project="ship",
        )
    )
    _strand_backup(registry, team.id)

    fresh = TeamRegistry(tmp_path)
    listed = fresh.list_teams()
    shown = fresh.get_team_by_name("survivor")
    assert [row.name for row in listed] == ["survivor"]
    assert shown is not None
    assert (
        shown.description,
        shown.manager,
        [(member.role, member.count) for member in shown.members],
        shown.instructions,
        shown.project,
    ) == ("durable", "lead", [("coder", 2)], "collaborate", "ship")
    assert not any(".backup." in path.name for path in registry.teams_dir.iterdir())


@pytest.mark.parametrize("command", [("list",), ("show", "survivor")])
def test_real_cli_read_recovers_stranded_backup(tmp_path: Path, command: tuple[str, ...]) -> None:
    config_dir = tmp_path / "config"
    registry = TeamRegistry(config_dir)
    team = registry.create_team(
        TeamEditFields(
            name="survivor",
            manager="lead",
            members=[TeamMember(role="coder")],
            instructions="collaborate",
            project="ship",
        )
    )
    _strand_backup(registry, team.id)

    result = _run_teams_cli(config_dir, *command)

    assert result.returncode == 0, result.stderr
    assert "survivor" in result.stdout
    assert "No teams found" not in result.stdout
    assert "No team found" not in result.stdout
    if command[0] == "show":
        assert "lead" in result.stdout
        assert "collaborate" in result.stdout
        assert "ship" in result.stdout


@pytest.mark.parametrize("operation", ["create", "update", "delete", "list", "show"])
def test_failed_recovery_aborts_every_registry_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="ops", instructions="old"))
    backup = _strand_backup(registry, team.id)
    real_replace = teams_module.os.replace

    def deny_restore(src, dst, **kwargs):
        if Path(src) == backup:
            raise PermissionError("denied")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(teams_module.os, "replace", deny_restore)
    with pytest.raises(TeamRegistryRecoveryError, match="fix registry permissions and retry"):
        if operation == "create":
            registry.create_team(TeamEditFields(name="ops"))
        elif operation == "update":
            registry.update_team(team.id, TeamEditFields(description="new"))
        elif operation == "delete":
            registry.delete_team(team.id)
        elif operation == "list":
            registry.list_teams()
        else:
            registry.get_team_by_name("ops")

    assert backup.is_dir()
    assert not (registry.teams_dir / team.id).exists()
    monkeypatch.undo()
    recovered = TeamRegistry(tmp_path).get_team_by_name("ops")
    assert recovered is not None
    assert (recovered.instructions, recovered.description) == ("old", "")


def test_recovery_fsync_failure_reports_then_row_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="survivor", instructions="old"))
    _strand_backup(registry, team.id)
    real_fsync_dir = teams_module._fsync_dir
    calls = 0

    def fail_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EIO, "durability failure")
        real_fsync_dir(path)

    monkeypatch.setattr(teams_module, "_fsync_dir", fail_once)
    with pytest.raises(TeamRegistryRecoveryError):
        registry.list_teams()
    monkeypatch.undo()

    recovered = TeamRegistry(tmp_path).get_team_by_name("survivor")
    assert recovered is not None
    assert recovered.instructions == "old"


def test_real_cli_failed_recovery_prints_guidance_not_not_found(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    registry = TeamRegistry(config_dir)
    team = registry.create_team(TeamEditFields(name="survivor"))
    backup = _strand_backup(registry, team.id)
    # The subprocess runs as the same user, so use a non-directory target that
    # makes backup->target fail deterministically rather than relying on chmod.
    (registry.teams_dir / team.id).write_text("blocks restore", encoding="utf-8")

    result = _run_teams_cli(config_dir, "show", "survivor")

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Could not recover team" in result.stderr
    assert "fix registry permissions and retry" in result.stderr
    assert "No team found" not in result.stderr
    assert backup.is_dir()


def test_lock_timeout_is_a_domain_exception_with_guidance(tmp_path: Path) -> None:
    """U5-1: the timeout carries retry guidance and a catchable narrow type."""
    registry = TeamRegistry(tmp_path)
    registry.create_team(TeamEditFields(name="seed"))

    holder = TeamRegistry(tmp_path)
    with holder._persistence_lock():
        with pytest.raises(TeamRegistryLockTimeout) as excinfo:
            registry.create_team(TeamEditFields(name="blocked"))
    message = str(excinfo.value)
    assert "Timed out waiting for the teams registry lock" in message
    assert "retry after the other lop process finishes" in message
    assert isinstance(excinfo.value, TimeoutError)


def test_cli_lock_timeout_prints_one_line_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """U5-1: the CLI presents registry contention as recoverable, not a crash.

    A peer process holds the lock past the bounded wait. The supported
    ``lop teams create`` path must exit non-zero with the retry guidance and
    WITHOUT the stack-trace panel the generic handler prints.
    """
    from local_operator.cli import main as cli_main

    cfg = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))

    holder = TeamRegistry(cfg)
    holder.create_team(TeamEditFields(name="seed"))  # creates the lock sidecar

    # Hold the lock from a raw fd so the CLI's own registry cannot acquire it.
    import fcntl

    lock_path = cfg / ".teams.lock"
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        # Shrink the bounded wait so the test is fast; the surface under test
        # (message + no traceback) is independent of the wait duration.
        monkeypatch.setattr("local_operator.teams._TEAM_LOCK_TIMEOUT_S", 0.2)
        monkeypatch.setattr("sys.argv", ["local-operator", "teams", "create", "blocked-squad"])
        code = cli_main()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert code == 1
    captured = capsys.readouterr()
    assert "Timed out waiting for the teams registry lock" in captured.err
    assert "retry after the other lop process finishes" in captured.err
    assert "Stack Trace" not in captured.err
    assert "Traceback (most recent call last)" not in captured.err
    assert captured.out == ""


def test_concurrent_readers_never_see_mixed_row_during_repeated_updates(
    tmp_path: Path,
) -> None:
    """R5-2: unlocked readers observe old-complete or new-complete, never mixed.

    One writer flips a team between two complete revisions (both fields change
    together) while several readers refresh continuously. Every observation
    must be one of the two complete tuples; a file-by-file writer would show
    (new-name, old-brief) mixtures.
    """
    writer = TeamRegistry(tmp_path)
    team = writer.create_team(
        TeamEditFields(name="rev-a", instructions="brief-a", project="proj-a")
    )
    revisions = [
        ("rev-a", "brief-a", "proj-a"),
        ("rev-b", "brief-b", "proj-b"),
    ]
    stop = threading.Event()
    mixed: list[tuple[str, str, str]] = []
    observations = [0]
    gaps = [0]

    def reader_loop() -> None:
        reader = TeamRegistry(tmp_path, refresh_interval=0)
        while not stop.is_set():
            for row in reader.list_teams():
                observations[0] += 1
                # Hydrate briefs the way a real attach does. A reader hitting
                # the exact swap gap misses the row entirely — the documented
                # tiny window with no target — so a transient KeyError is a
                # legitimate observation. The INVARIANT under test is that no
                # observation is a MIXED revision (new metadata with old or
                # empty briefs), which a file-by-file writer would produce.
                try:
                    full = reader.get_team(row.id)
                except KeyError:
                    gaps[0] += 1
                    continue
                triple = (full.name, full.instructions, full.project)
                if triple not in revisions:
                    mixed.append(triple)

    readers = [threading.Thread(target=reader_loop) for _ in range(3)]
    for r in readers:
        r.start()
    try:
        for flip in range(40):
            target = revisions[flip % 2]
            writer.update_team(
                team.id,
                TeamEditFields(name=target[0], instructions=target[1], project=target[2]),
            )
    finally:
        stop.set()
        for r in readers:
            r.join(timeout=10)
    assert all(not r.is_alive() for r in readers)
    # The writer finished on the last revision it wrote.
    assert observations[0] > 0, "readers never observed the registry"
    assert mixed == [], f"mixed revisions observed: {mixed[:5]}"
    final = TeamRegistry(tmp_path).get_team(team.id)
    assert (final.name, final.instructions, final.project) in revisions


def _hold_lock_process(config_dir: Path, seconds: float) -> subprocess.Popen[str]:
    """Hold the registry lock from ANOTHER process, as a peer `lop` does."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys, time\n"
                "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                "sys.stdout.write('held\\n'); sys.stdout.flush()\n"
                "time.sleep(float(sys.argv[2]))\n"
            ),
            str(config_dir / ".teams.lock"),
            str(seconds),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "held", "peer never took the lock"
    return proc


def test_read_never_waits_on_a_peer_holding_the_lock(tmp_path: Path) -> None:
    """R7-1: a contended read serves the current view instead of blocking.

    The reported freeze: `_team_choices` calls `list_teams` synchronously on the
    TUI event loop, and a crash artifact made every read take the bounded 10s
    WRITER wait. One `/team ` keystroke fans out into several reads, so a peer
    mid-publish froze the whole app for a minute. The read path must try-acquire
    and move on.
    """
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="visible"))
    registry.list_teams()
    _strand_backup(registry, team.id)

    holder = _hold_lock_process(tmp_path, 30)
    try:
        started = time.monotonic()
        for _ in range(6):  # one keystroke's fan-out
            registry._last_refresh_time = 0.0
            registry.list_teams()
        elapsed = time.monotonic() - started
    finally:
        holder.kill()
        holder.wait()

    # Generous bound: the point is orders of magnitude, not a tight budget. The
    # unfixed path took 6 x 10s here; anything near a second is a regression.
    assert elapsed < 1.0, f"contended reads blocked for {elapsed:.2f}s"


def test_contended_read_recovers_once_the_peer_releases(tmp_path: Path) -> None:
    """R7-1: skipping recovery is a DEFERRAL, never a permanent give-up."""
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="survivor", instructions="briefs"))
    registry.list_teams()
    _strand_backup(registry, team.id)

    holder = _hold_lock_process(tmp_path, 30)
    try:
        registry._last_refresh_time = 0.0
        # Contended: the row is not recoverable right now, and the reader says
        # so by serving what it has rather than raising or hanging.
        registry.list_teams()
    finally:
        holder.kill()
        holder.wait()

    # The cooldown bounds ATTEMPTS, so a read past it retries and heals.
    registry._recovery_attempted_at = 0.0
    registry._last_refresh_time = 0.0
    names = [t.name for t in registry.list_teams()]
    assert names == ["survivor"]
    assert (registry.teams_dir / team.id).is_dir()


def test_repeated_reads_make_at_most_one_recovery_attempt_per_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7-1: many rapid reads must not each pay a recovery probe."""
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="ops"))
    registry.list_teams()
    _strand_backup(registry, team.id)

    attempts = 0
    real_lock = TeamRegistry._persistence_lock

    def counting_lock(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        return real_lock(self, *args, **kwargs)

    monkeypatch.setattr(TeamRegistry, "_persistence_lock", counting_lock)
    holder = _hold_lock_process(tmp_path, 20)
    try:
        for _ in range(24):  # four keystrokes' worth of reads
            registry._last_refresh_time = 0.0
            registry.list_teams()
    finally:
        holder.kill()
        holder.wait()
    assert attempts == 1, f"{attempts} recovery attempts for one unchanged artifact"


def test_failed_restore_keeps_raising_rather_than_reporting_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6-4 must survive R7-1: a real restore FAILURE is not softened.

    The cooldown that bounds contended attempts must never turn an artifact
    that genuinely cannot be restored into a silent "no teams" — that is the
    U6-1 failure mode, and the row is still on disk.
    """
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="survivor"))
    backup = _strand_backup(registry, team.id)
    real_replace = teams_module.os.replace

    def deny(src, dst, **kwargs):
        if Path(src) == backup:
            raise PermissionError("denied")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(teams_module.os, "replace", deny)
    # EVERY read keeps raising, including ones inside the attempt cooldown.
    for _ in range(5):
        registry._last_refresh_time = 0.0
        with pytest.raises(TeamRegistryRecoveryError, match="fix registry permissions"):
            registry.list_teams()
    assert backup.is_dir()


def test_construction_never_raises_and_defers_the_error_to_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7-2: a broken registry must not abort the session that constructs it.

    `session_factory` builds the registry during boot, before the model, the
    tools and the transcript exist, so a raise there took down the whole app
    over a `teams/` directory the user may never have used.
    """
    registry = TeamRegistry(tmp_path)
    team = registry.create_team(TeamEditFields(name="survivor"))
    backup = _strand_backup(registry, team.id)
    real_replace = teams_module.os.replace

    def deny(src, dst, **kwargs):
        if Path(src) == backup:
            raise PermissionError("denied")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(teams_module.os, "replace", deny)
    fresh = TeamRegistry(tmp_path)  # must not raise
    assert isinstance(fresh.recovery_error, TeamRegistryRecoveryError)
    # ...but the first real read still refuses to answer with a half-truth.
    with pytest.raises(TeamRegistryRecoveryError):
        fresh.list_teams()

    monkeypatch.undo()
    recovered = TeamRegistry(tmp_path).get_team_by_name("survivor")
    assert recovered is not None
