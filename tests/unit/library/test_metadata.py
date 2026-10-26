from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from local_operator.library.metadata import (
    MAX_TAGS_PER_ASSET,
    SCHEMA_VERSION,
    Attribution,
    MetadataStore,
    MetadataUnavailable,
    TagRefused,
    reset_schema_readiness,
)

A = "la_0123456789abcdef"
B = "la_fedcba9876543210"


def store(tmp_path: Path) -> MetadataStore:
    """A store in this test's own scratch root.

    The process-wide schema-readiness cache is cleared first: pytest hands two
    tests the same ``tmp_path`` when their names truncate to the same 30
    characters, and a recycled directory holds no schema even though the cache
    still claims one (measured: a 0-byte store and "no such table: assets").
    """
    reset_schema_readiness()
    return MetadataStore(tmp_path / "cfg")


def test_schema_and_asset_provenance(tmp_path) -> None:
    s = store(tmp_path)
    row = s.ensure_asset(A, "p:/tmp/a.png", derived_name="a.png")
    assert row and row["name_source"] == "derived"
    assert s.schema_version() == SCHEMA_VERSION

    updated = s.update_fields(A, {"name": "Cat", "description": "A cat"}, actor="librarian")
    assert updated and updated["name_source"] == "librarian"
    # A later derived refresh must not overwrite the librarian's field.
    s.ensure_asset(A, "p:/tmp/a.png", derived_name="renamed-on-disk.png")
    row = s.asset(A)
    assert row is not None and row["name"] == "Cat"

    user = s.update_fields(A, {"name": "My cat"}, actor="user")
    assert user is not None and user["name_source"] == "user"
    s.ensure_asset(A, "p:/tmp/a.png", derived_name="again.png")
    row = s.asset(A)
    assert row is not None and row["name"] == "My cat"
    edits = s.edits(A)
    assert len(edits) == 2 and edits[0]["actor"] == "user"


def test_tags_primary_limit_and_merge(tmp_path) -> None:
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    s.ensure_asset(B, "p:/b")
    travel = s.create_tag("Travel")
    assert travel["id"] == s.create_tag("travel")["id"], "case-insensitive, idempotent"
    work = s.create_tag("Work", "project", actor="librarian")

    s.set_asset_tags(A, [travel["id"], work["id"]], primary=travel["id"], actor="user")
    assert s.primary_tags([A])[A] == travel["id"]
    assert {t["name"] for t in s.asset_tags(A)} == {"travel", "work"}

    with pytest.raises(TagRefused):
        s.set_asset_tags(A, [travel["id"]], primary=work["id"], actor="user")
    with pytest.raises(TagRefused):
        s.set_asset_tags(A, [f"{i:010x}" for i in range(MAX_TAGS_PER_ASSET + 1)], actor="user")
    with pytest.raises(TagRefused):
        s.set_asset_tags(A, ["deadbeef00"], actor="user")

    s.set_asset_tags(B, [travel["id"]], actor="user")
    merged = s.merge_tags(travel["id"], work["id"])
    assert merged["id"] == work["id"]
    assert {t["name"]: t["count"] for t in s.list_tags()}["work"] == 2
    assert s.tag_by_id(travel["id"]) is None
    # A primary pointing at the merged-away tag follows to the survivor.
    assert s.primary_tags([A])[A] == work["id"]

    renamed = s.rename_tag(work["id"], "job")
    assert renamed["name"] == "job"
    assert {t["name"] for t in s.asset_tags(A)} == {"job"}
    assert s.delete_tag(work["id"]) is True
    assert s.asset_tags(A) == []


def test_conventions_chat_and_edit_history(tmp_path) -> None:
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    first = s.set_convention("naming", "naming", "Title case")
    assert first["body"] == "Title case"
    s.set_convention("naming", "naming", "Sentence case", actor="user")
    assert s.conventions()[0]["body"] == "Sentence case"

    s.set_chat(A, "abcdef123456")
    chat = s.chat(A)
    assert chat is not None and chat["session_id"] == "abcdef123456"
    assert s.clear_chat(A) is True
    assert s.chat(A) is None

    s.update_fields(A, {"favorite": True}, actor="user")
    assert s.assets([A])[A]["favorite"] == 1
    assert s.stats()["assets"] == 1


def test_a_future_or_corrupt_database_answers_an_empty_catalog(tmp_path) -> None:
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a", derived_name="a")
    with s._write() as connection:
        connection.execute("UPDATE state SET value = '99' WHERE key = 'schema_version'")
    # A NEW PROCESS is what meets a future store, not a reader that has already
    # verified the schema; the process-wide readiness cache is cleared to model
    # that boundary (production never re-learns inside one process).
    reset_schema_readiness()
    future = store(tmp_path)
    assert future.asset(A) is None
    assert future.list_tags() == []
    assert future.conventions() == []
    assert future.stats()["available"] is False
    # Reads degrade to empty; a WRITE that cannot be recorded raises so the
    # route can refuse instead of pretending the edit landed.
    assert future.ensure_asset(B, "p:/b") is None
    with pytest.raises(MetadataUnavailable):
        future.update_fields(B, {"favorite": True}, actor="user")


def test_concurrent_writes_do_not_tear_or_deadlock(tmp_path) -> None:
    s = store(tmp_path)

    def write(index: int) -> None:
        asset = f"la_{index:016x}"
        s.ensure_asset(asset, f"p:/tmp/{index}")
        tag = s.create_tag(f"tag-{index}")
        s.set_asset_tags(asset, [tag["id"]], actor="user")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(24)))
    assert s.stats()["assets"] == 24
    assert len(s.list_tags()) == 24


def test_mark_organized_stamps_in_the_tag_transaction(tmp_path) -> None:
    """A tag set the caller calls organizing stamps inside the SAME write.

    The asset's own ``confidence`` stays NULL: tag confidence lives on
    ``asset_tags``, and an organized tag row is not a confidence about the
    asset. No second ``edits`` row is written either — the tag set already has
    one, and a stamp is not a second edit by a second writer.
    """
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    travel = s.create_tag("Travel")
    fresh = s.asset(A)
    assert fresh is not None and fresh["organized_at"] is None

    s.set_asset_tags(A, [travel["id"]], actor="librarian", mark_organized=True)
    row = s.asset(A)
    assert row is not None
    assert row["organized_by"] == "librarian" and row["organized_at"]
    assert row["confidence"] is None
    assert len(s.edits(A)) == 1

    # A CLEAR is not an un-organization: the stamp survives the empty set.
    s.set_asset_tags(A, [], actor="librarian", mark_organized=False)
    cleared = s.asset(A)
    assert cleared is not None
    assert cleared["organized_at"] == row["organized_at"]
    assert s.asset_tags(A) == []


def test_apply_attributions_moves_the_rows_and_sets_the_guard(tmp_path) -> None:
    """One transaction: provenance, the stamp, the tags, and the guard fact."""
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    s.ensure_asset(B, "p:/b")
    edit = s.update_fields(A, {"name": "Hyundai re-screen"}, actor="agent")
    assert edit is not None
    tag = s.create_tag("hyundai-poc")
    s.set_asset_tags(A, [tag["id"]], actor="agent")
    revision = s.metadata_revision()

    moved = s.apply_attributions(
        [Attribution(A, ("name",), 1234.5, tag_ids=(tag["id"],))],
        guard_key="backfill:test:v1",
        guard_value='{"matched":1}',
    )
    assert moved == 1
    row = s.asset(A)
    assert row is not None
    assert row["name_source"] == "librarian" and row["name"] == "Hyundai re-screen"
    assert row["organized_at"] == 1234.5 and row["organized_by"] == "librarian"
    assert [t["source"] for t in s.asset_tags(A)] == ["librarian"]
    # The pass DID move metadata, so a cursor minted before it must be refused.
    assert s.metadata_revision() != revision
    assert s.state_value("backfill:test:v1") == '{"matched":1}'
    # The audit log is untouched: this pass re-read it rather than adding to it.
    assert len(s.edits(A)) == 2

    # A row that is already organized cannot be stamped twice, and an empty
    # pass still records its guard.
    assert (
        s.apply_attributions(
            [Attribution(A, ("name",), 9999.0, tag_ids=(tag["id"],))],
            guard_key="backfill:test:v1",
            guard_value='{"matched":0}',
        )
        == 0
    )
    again = s.asset(A)
    assert again is not None and again["organized_at"] == 1234.5


def test_apply_attributions_only_moves_the_tag_ids_it_is_given(tmp_path) -> None:
    """A tag row moves by ID, so a set someone else applied stays theirs.

    ``asset_tags`` holds a set, so a later write IS a replacement of hers (a
    delete plus re-insert, which is how the column is written). Only the id the
    evidence carries may move; every other row on the asset keeps its own
    provenance. The guard key is opaque to this test: only the rows matter.
    """
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    s.update_fields(A, {"name": "Hyundai re-screen"}, actor="agent")
    hers = s.create_tag("hyundai-poc")
    theirs = s.create_tag("eng-3631")
    s.set_asset_tags(A, [hers["id"], theirs["id"]], actor="agent")

    assert (
        s.apply_attributions(
            [Attribution(A, ("name",), 1234.5, tag_ids=(hers["id"],))],
            guard_key="scratch-pass",
            guard_value='{"matched":1}',
        )
        == 1
    )
    assert {tag["id"]: tag["source"] for tag in s.asset_tags(A)} == {
        hers["id"]: "librarian",
        theirs["id"]: "agent",
    }


def test_apply_attributions_without_tag_ids_leaves_every_tag_alone(tmp_path) -> None:
    """No tag evidence means no tag row moves — not "all of them"."""
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    s.update_fields(A, {"name": "Hyundai re-screen"}, actor="agent")
    tag = s.create_tag("hyundai-poc")
    s.set_asset_tags(A, [tag["id"]], actor="agent")

    assert (
        s.apply_attributions(
            [Attribution(A, ("name",), 1234.5)],
            guard_key="scratch-pass",
            guard_value='{"matched":1}',
        )
        == 1
    )
    assert [row["source"] for row in s.asset_tags(A)] == ["agent"]


def test_apply_attributions_leaves_locked_and_user_rows_alone(tmp_path) -> None:
    """The candidate read and the write agree on who may be moved."""
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    s.ensure_asset(B, "p:/b")
    s.update_fields(A, {"name": "Locked"}, actor="agent")
    s.update_fields(A, {"locked": True}, actor="user")
    s.update_fields(B, {"name": "Mine"}, actor="user")

    candidates = [row["asset_id"] for row in s.unorganized_agent_assets()]
    assert candidates == []
    assert (
        s.apply_attributions(
            [Attribution(A, ("name",), 5.0), Attribution(B, ("name",), 5.0)],
            guard_key="backfill:test:locked",
            guard_value="{}",
        )
        == 0
    )
    for asset in (A, B):
        row = s.asset(asset)
        assert row is not None and row["organized_at"] is None


def test_the_candidate_read_is_the_re_attributions_predicate(tmp_path) -> None:
    """Only agent-owned, unlocked, unorganized rows are candidates."""
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a", derived_name="a.png")
    s.ensure_asset(B, "p:/b")
    s.update_fields(B, {"name": "Hers"}, actor="librarian")
    assert [row["asset_id"] for row in s.unorganized_agent_assets()] == []
    s.update_fields(A, {"description": "An agent's words"}, actor="agent")
    assert [row["asset_id"] for row in s.unorganized_agent_assets()] == [A]
    s.update_fields(A, {"organized_at": 1.0, "organized_by": "librarian"}, actor="librarian")
    assert s.unorganized_agent_assets() == []


def test_last_edit_at_reads_the_newest_row_by_actor(tmp_path) -> None:
    """The re-attribution dates an organization to the edit that made it."""
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    assert s.last_edit_at(A) is None
    s.update_fields(A, {"name": "First"}, actor="agent", at=100.0)
    s.update_fields(A, {"name": "Second"}, actor="agent", at=200.0)
    s.update_fields(A, {"name": "Mine"}, actor="user", at=300.0)
    assert s.last_edit_at(A) == 300.0
    assert s.last_edit_at(A, actor="agent") == 200.0
    assert s.last_edit_at(A, actor="librarian") is None


def test_last_edit_at_can_narrow_to_the_edits_that_touched_a_field(tmp_path) -> None:
    """``fields`` is what makes the stamp her naming edit, not a later edit.

    A favorite and a tag set are both ordinary agent edits on the same asset,
    and neither of them organized it.
    """
    s = store(tmp_path)
    s.ensure_asset(A, "p:/a")
    s.update_fields(A, {"name": "First"}, actor="agent", at=100.0)
    s.update_fields(A, {"favorite": True}, actor="agent", at=200.0)
    tag = s.create_tag("hyundai-poc")
    s.set_asset_tags(A, [tag["id"]], actor="agent")

    newest = s.last_edit_at(A, actor="agent")
    assert newest is not None and newest > 200.0
    assert s.last_edit_at(A, actor="agent", fields=("name", "description")) == 100.0
    # The narrowing filters the SAME rows: it is not a second clock.
    assert s.last_edit_at(A, actor="agent", fields=("favorite",)) == 200.0
    assert s.last_edit_at(A, actor="user", fields=("name",)) is None
    # An edit row whose ``fields`` column cannot be parsed is skipped, not fatal.
    with sqlite3.connect(s.path) as connection:
        connection.execute("UPDATE edits SET fields = 'not json' WHERE asset_id = ?", (A,))
    assert s.last_edit_at(A, actor="agent", fields=("name",)) is None
    unfiltered = s.last_edit_at(A, actor="agent")
    assert unfiltered is not None and unfiltered > 200.0
