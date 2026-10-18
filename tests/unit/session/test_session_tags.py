"""The durable registry and assignment contract, without a server/runtime."""

import json
from pathlib import Path

import pytest

from local_operator.session.session_tags import (
    read_categories,
    read_entity_tags,
    resolve_category_names,
    set_entity_tags,
    usage_counts,
    write_categories,
)

SID = "abcdef123456"


def test_category_lifecycle_and_assignment_identity(tmp_path: Path) -> None:
    a, b = write_categories(tmp_path, [(None, " Review "), (None, "Fix")])
    assert resolve_category_names(tmp_path, ["review", " FIX "]) == [a["id"], b["id"]]
    set_entity_tags(tmp_path, "session", SID, [a["id"], b["id"]], a["id"])
    assert usage_counts(tmp_path) == {a["id"]: 1, b["id"]: 1}
    assert write_categories(tmp_path, [(b["id"], "Fixes"), (a["id"], "Review")])[0]["id"] == b["id"]
    assert read_entity_tags(tmp_path, "session", SID)["primary"] == a["id"]
    write_categories(tmp_path, [(b["id"], "Fixes")])
    assert read_entity_tags(tmp_path, "session", SID) == {"tags": [b["id"]], "primary": b["id"]}
    recreated = write_categories(tmp_path, [(b["id"], "Fixes"), (None, "Review")])[1]
    assert recreated["id"] != a["id"]
    assert read_entity_tags(tmp_path, "session", SID)["tags"] == [b["id"]]


@pytest.mark.parametrize(
    "desired",
    [
        [(None, "")],
        [(None, "x" * 25)],
        [(None, " unTAGGED ")],
        [(None, "Review"), (None, "review")],
        [("aaaaaaaaaa", "Unknown")],
        [(None, str(i)) for i in range(33)],
    ],
)
def test_bad_registry_is_atomic(tmp_path: Path, desired) -> None:
    before = write_categories(tmp_path, [(None, "Original")])
    with pytest.raises(ValueError):
        write_categories(tmp_path, desired)
    assert read_categories(tmp_path) == before


def test_duplicate_id_and_caps(tmp_path: Path) -> None:
    categories = write_categories(tmp_path, [(None, str(i)) for i in range(32)])
    key = categories[0]["id"]
    with pytest.raises(ValueError):
        write_categories(tmp_path, [(key, "One"), (key, "Two")])
    keys = [tag["id"] for tag in categories]
    set_entity_tags(tmp_path, "project", "project-1", keys[:8], keys[0])
    with pytest.raises(ValueError):
        set_entity_tags(tmp_path, "session", SID, keys[:9], keys[0])
    with pytest.raises(ValueError):
        set_entity_tags(tmp_path, "session", SID, keys[:1], keys[1])
    with pytest.raises(ValueError):
        set_entity_tags(tmp_path, "session", SID, ["unknown"], None)
    with pytest.raises(ValueError):
        set_entity_tags(tmp_path, "session", SID, [key, key], key)


@pytest.mark.parametrize(
    "kind,key",
    [
        ("other", SID),
        ("session", "../escape"),
        ("session", "/absolute"),
        ("session", ".."),
        ("session", "."),
        ("session", "not-an-id"),
        ("project", "../escape"),
        ("project", "."),
        ("project", "a/b"),
    ],
)
def test_entity_paths_are_validated(tmp_path: Path, kind: str, key: str) -> None:
    with pytest.raises(ValueError):
        read_entity_tags(tmp_path, kind, key)
    with pytest.raises(ValueError):
        set_entity_tags(tmp_path, kind, key, [], None)


def test_absent_corrupt_and_invalid_primary(tmp_path: Path) -> None:
    empty = {"tags": [], "primary": None}
    assert read_categories(tmp_path) == []
    assert read_entity_tags(tmp_path, "session", SID) == empty
    a, b = write_categories(tmp_path, [(None, "A"), (None, "B")])
    set_entity_tags(tmp_path, "session", SID, [b["id"], a["id"]], None)
    assert read_entity_tags(tmp_path, "session", SID)["primary"] == b["id"]
    path = tmp_path / "tags" / "session" / f"{SID}.json"
    path.write_text(
        json.dumps({"version": 1, "tags": [b["id"], "orphan", a["id"]], "primary": "bad"})
    )
    assert read_entity_tags(tmp_path, "session", SID) == {
        "tags": [b["id"], a["id"]],
        "primary": b["id"],
    }
    for contents in ("broken", "[]", '{"version":1,"tags":null}', '{"version":2,"tags":[]}'):
        path.write_text(contents)
        assert read_entity_tags(tmp_path, "session", SID) == empty
    (tmp_path / "tags" / "categories.json").write_text("broken")
    assert read_categories(tmp_path) == []


def test_resolver_refuses_unknown_blank_duplicate_and_limits(tmp_path: Path) -> None:
    write_categories(tmp_path, [(None, "Review")])
    with pytest.raises(ValueError, match="Known categories: Review"):
        resolve_category_names(tmp_path, ["Typo"])
    for names in ([], [" "], ["Review"] * 9, ["Review", "review"]):
        with pytest.raises(ValueError):
            resolve_category_names(tmp_path, names)


def test_malformed_registry_entries_do_not_leak(tmp_path: Path) -> None:
    path = tmp_path / "tags" / "categories.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "categories": [
                    None,
                    {"id": "aaaaaaaaaa", "name": " Untagged "},
                    {"id": "bbbbbbbbbb", "name": "Valid"},
                    {"id": "cccccccccc", "name": " valid "},
                    {"id": "../../outside", "name": "Invalid"},
                ],
            }
        )
    )
    assert read_categories(tmp_path) == [{"id": "bbbbbbbbbb", "name": "Valid"}]
