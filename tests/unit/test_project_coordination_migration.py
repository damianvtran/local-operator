"""The schema-1 -> 2 coordination migration: plan, apply, idempotence.

The migration exists because the chief of staff's create-time auto-link wrote
her session id into ``sessions`` — the WORK set — so 30 rows on the operator's
store count her as a worker and earn her a completion-check nudge for projects
she filed but does not work on. It is PRESERVATIVE (re-kind, never delete),
runs from ONE startup seam, backs up every touched row before any rewrite, and
its gate is an idempotent predicate — no stamp file (the config-migrations
doctrine; see :mod:`local_operator.config_migrations`).

These tests own the fixture store, the golden dry-run table, and the four
operational guarantees: dry runs write nothing (not even a lock), apply backs
up before rewriting, a second apply is a no-op, and both a held lock and the
schema guard leave rows byte-unchanged.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

import local_operator.projects as projects
from local_operator.projects import (
    PROJECT_SCHEMA,
    ProjectEdit,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    migrate_coordination_links,
)

COS = "439818272d84"
WORKER = "4e92693767fa"
OTHER = "abcdef012345"
COS_NAME = "Aida"


def _row(
    project_id: str,
    name: str,
    *,
    sessions: list[str],
    coordination: list[str] | None = None,
    schema: int = 1,
    owner: str | None = None,
    updates: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """One schema-1-shaped row; only the fields this migration reads are set."""
    return {
        "id": project_id,
        "name": name,
        "schema": schema,
        "status": "active",
        "owner": owner,
        "sessions": list(sessions),
        "coordination_sessions": list(coordination or []),
        "updates": list(updates or []),
        "created_at": 1.0,
        "updated_at": 1.0,
    }


def _write_row(root: Path, payload: dict[str, object]) -> None:
    (root / "projects").mkdir(parents=True, exist_ok=True)
    (root / "projects" / f"{payload['id']}.json").write_text(json.dumps(payload))


def _fixture_store(root: Path) -> None:
    """Eight rows: four keep/demote shapes, one unplanned, one already migrated.

    The four planned outcomes cover every keep clause and the two demote
    shapes (no clause; a stale ``owner`` from before a rename — the caveat the
    ruling's dry run is FOR).
    """
    _write_row(
        root,
        _row(
            "0a0000000001",
            "authored-kept",
            sessions=[COS, WORKER],
            updates=[{"by": COS, "text": "one"}, {"by": COS, "text": "two"}],
        ),
    )
    _write_row(
        root,
        _row(
            "0a0000000002",
            "authored-mixed",
            sessions=[COS, WORKER],
            updates=[{"by": COS, "text": "one"}, {"by": WORKER, "text": "two"}],
        ),
    )
    _write_row(root, _row("0a0000000003", "demote-basic", sessions=[COS, WORKER]))
    _write_row(
        root,
        _row(
            "0a0000000004",
            "renamed-owner",
            sessions=[COS, WORKER],
            owner="Previous Name",
        ),
    )
    _write_row(root, _row("0a0000000005", "sole-kept", sessions=[COS]))
    _write_row(root, _row("0a0000000006", "upper-owner", sessions=[COS, WORKER], owner="AIDA"))
    _write_row(root, _row("0a0000000007", "other-session", sessions=[WORKER]))
    _write_row(
        root,
        _row(
            "0a0000000008",
            "already-migrated",
            schema=2,
            sessions=[WORKER],
            coordination=[COS],
        ),
    )


#: ``(id, name, decision, rule)`` for every row the selector matches, in plan
#: order (name-casefold sorted). 6 of 8 rows: the other two are unplanned.
GOLDEN = [
    ("0a0000000001", "authored-kept", "keep", "authored_all"),
    ("0a0000000002", "authored-mixed", "demote", "no_keep_clause"),
    ("0a0000000003", "demote-basic", "demote", "no_keep_clause"),
    ("0a0000000004", "renamed-owner", "demote", "no_keep_clause"),
    ("0a0000000005", "sole-kept", "keep", "sole_link"),
    ("0a0000000006", "upper-owner", "keep", "owner"),
]

KEPT = {"0a0000000001", "0a0000000005", "0a0000000006"}
DEMOTED = {"0a0000000002", "0a0000000003", "0a0000000004"}


def _plan_rows(plan: list[dict[str, object]]) -> list[tuple[object, ...]]:
    return [(row["id"], row["name"], row["decision"], row["rule"]) for row in plan]


def _bytes_by_id(root: Path) -> dict[str, bytes]:
    return {path.stem: path.read_bytes() for path in (root / "projects").glob("*.json")}


def test_the_dry_run_plan_matches_the_golden_table_and_writes_nothing(tmp_path: Path) -> None:
    _fixture_store(tmp_path)
    before = _bytes_by_id(tmp_path)

    plan = migrate_coordination_links(
        tmp_path, dry_run=True, cos_session_id=COS, cos_display_name=COS_NAME
    )

    assert _plan_rows(plan) == GOLDEN
    # The planned lists are on the plan, per row.
    demote = next(row for row in plan if row["id"] == "0a0000000003")
    assert demote["sessions"] == [WORKER]
    assert demote["coordination_sessions"] == [COS]
    keep = next(row for row in plan if row["id"] == "0a0000000005")
    assert keep["sessions"] == [COS] and keep["coordination_sessions"] == []
    # NOTHING was written: no row bytes moved, no backup dir, and not even the
    # store lock — a dry run is a read.
    assert _bytes_by_id(tmp_path) == before
    assert not list((tmp_path / "projects").glob(".migrations-backup-*"))
    assert not (tmp_path / "projects" / ".lock").exists()


def test_apply_re_kinds_only_the_demoted_rows_and_writes_then_schema_2(
    tmp_path: Path,
) -> None:
    _fixture_store(tmp_path)
    before = _bytes_by_id(tmp_path)

    plan = migrate_coordination_links(tmp_path, cos_session_id=COS, cos_display_name=COS_NAME)
    assert _plan_rows(plan) == GOLDEN

    for project_id in DEMOTED:
        row = json.loads((tmp_path / "projects" / f"{project_id}.json").read_text())
        assert row["sessions"] == [WORKER], project_id
        assert row["coordination_sessions"] == [COS], project_id
        assert row["schema"] == PROJECT_SCHEMA, project_id
    for project_id in KEPT:
        row = json.loads((tmp_path / "projects" / f"{project_id}.json").read_text())
        # A wrong keep leaves the row as today: same schema, same lists.
        assert row["schema"] == 1, project_id
        assert COS in row["sessions"], project_id
    for project_id in ("0a0000000007", "0a0000000008"):
        # Unplanned rows are not rewritten at all — byte-identical, whatever
        # schema they already carried (7 is schema 1, 8 is an earlier v2 row).
        assert (tmp_path / "projects" / f"{project_id}.json").read_bytes() == before[
            project_id
        ], project_id

    # Backup FIRST, and byte-exact: the demoted rows' original bytes live in
    # one stamp dir; nothing else was copied.
    backups = list((tmp_path / "projects").glob(".migrations-backup-*"))
    assert len(backups) == 1
    backed = {path.stem: path.read_bytes() for path in backups[0].glob("*.json")}
    assert set(backed) == DEMOTED
    for project_id, payload in backed.items():
        assert payload == before[project_id]


def test_a_second_apply_writes_nothing(tmp_path: Path) -> None:
    _fixture_store(tmp_path)
    migrate_coordination_links(tmp_path, cos_session_id=COS, cos_display_name=COS_NAME)
    after_first = _bytes_by_id(tmp_path)
    backups_one = list((tmp_path / "projects").glob(".migrations-backup-*"))

    plan = migrate_coordination_links(tmp_path, cos_session_id=COS, cos_display_name=COS_NAME)

    # The demotion predicate is exhausted; the kept rows re-decide to keep.
    assert [row["decision"] for row in plan] == ["keep"] * len(KEPT)
    assert _bytes_by_id(tmp_path) == after_first
    assert list((tmp_path / "projects").glob(".migrations-backup-*")) == backups_one


def test_no_chief_of_staff_means_nothing_to_plan(tmp_path: Path) -> None:
    _fixture_store(tmp_path)
    assert migrate_coordination_links(tmp_path, cos_session_id="", cos_display_name=COS_NAME) == []
    assert migrate_coordination_links(tmp_path, dry_run=True) == []  # no aida state either


def test_a_held_lock_refuses_and_leaves_every_row_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture_store(tmp_path)
    before = _bytes_by_id(tmp_path)
    other = ProjectRegistry(tmp_path)
    monkeypatch.setattr(projects, "_LOCK_TIMEOUT_S", 0.2)
    held = other._persistence_lock()
    held.__enter__()
    try:
        with pytest.raises(ProjectRegistryLockTimeout):
            migrate_coordination_links(tmp_path, cos_session_id=COS, cos_display_name=COS_NAME)
    finally:
        held.__exit__(None, None, None)
    assert _bytes_by_id(tmp_path) == before
    assert not list((tmp_path / "projects").glob(".migrations-backup-*"))


def test_an_unwritable_backup_aborts_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort-if-no-backup: a stamp dir that cannot be created leaves every row
    untouched (a later launch retries — nothing records the attempt as done)."""
    _fixture_store(tmp_path)
    before = _bytes_by_id(tmp_path)

    class _FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001, ANN206
            return datetime.datetime(2026, 9, 30, 12, 0, 0)

    monkeypatch.setattr(projects, "datetime", _FrozenDateTime)
    # A FILE where the backup dir must go makes mkdir raise (OSError).
    blocker = tmp_path / "projects" / ".migrations-backup-20260930-120000"
    blocker.write_text("not a directory")

    plan = migrate_coordination_links(tmp_path, cos_session_id=COS, cos_display_name=COS_NAME)

    assert _plan_rows(plan) == GOLDEN
    assert _bytes_by_id(tmp_path) == before


def test_a_migrated_row_refuses_mutation_from_a_simulated_older_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The schema bump's whole point: an older build's rewrite must refuse,
    not silently drop ``coordination_sessions``."""
    _fixture_store(tmp_path)
    migrate_coordination_links(tmp_path, cos_session_id=COS, cos_display_name=COS_NAME)
    monkeypatch.setattr(projects, "PROJECT_SCHEMA", 1)  # the older build
    store = ProjectRegistry(tmp_path)
    with pytest.raises(ProjectSchemaGuardError):
        store.update_project("0a0000000002", ProjectEdit(description="x"))
    row = json.loads((tmp_path / "projects" / "0a0000000002.json").read_text())
    assert row["description"] == ""
    assert row["coordination_sessions"] == [COS]


def test_the_startup_seam_applies_the_migration_once(tmp_path: Path) -> None:
    """``run_startup_migrations`` is the ONE seam (config-migrations doctrine):
    with aida state present it re-kinds the fixture store's rows."""
    from local_operator.aida.state import write_state
    from local_operator.config_migrations import run_startup_migrations

    _fixture_store(tmp_path)
    write_state(tmp_path, {"session_id": COS})
    run_startup_migrations(tmp_path)
    row = json.loads((tmp_path / "projects" / "0a0000000003.json").read_text())
    assert row["schema"] == PROJECT_SCHEMA
    assert row["coordination_sessions"] == [COS]

    # A second launch re-reads the exhausted predicate and changes nothing.
    after_first = _bytes_by_id(tmp_path)
    run_startup_migrations(tmp_path)
    assert _bytes_by_id(tmp_path) == after_first
