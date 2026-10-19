from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from local_operator.library.metadata import (
    MAX_TAGS_PER_ASSET,
    SCHEMA_VERSION,
    MetadataStore,
    MetadataUnavailable,
    TagRefused,
)

A = "la_0123456789abcdef"
B = "la_fedcba9876543210"


def store(tmp_path: Path) -> MetadataStore:
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
    # A NEW reader is the process that meets a future store, not one that has
    # already cached the schema as readable.
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
