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
    (tmp_path / "sessions" / SID).mkdir(parents=True)
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


def test_name_only_registry_retries_reuse_casefolded_identity(tmp_path: Path) -> None:
    original = write_categories(tmp_path, [(None, "Review"), (None, "Fix")])
    replayed = write_categories(tmp_path, [(None, " review "), (None, "FIX")])
    assert [tag["id"] for tag in replayed] == [tag["id"] for tag in original]
    draft: list[tuple[str | None, str]] = [(original[0]["id"], "Renamed"), (None, "review")]
    renamed, fresh = write_categories(tmp_path, draft)
    assert renamed == {"id": original[0]["id"], "name": "Renamed"}
    assert fresh["id"] not in {tag["id"] for tag in original}
    assert write_categories(tmp_path, draft) == [renamed, fresh]


def test_primary_changes_and_put_order_preserve_assignment_history(tmp_path: Path) -> None:
    a, b = write_categories(tmp_path, [(None, "A"), (None, "B")])
    set_entity_tags(tmp_path, "session", SID, [a["id"]], a["id"])
    state = set_entity_tags(tmp_path, "session", SID, [b["id"], a["id"]], b["id"])
    assert state == {"tags": [a["id"], b["id"]], "primary": b["id"]}
    state = set_entity_tags(tmp_path, "session", SID, [b["id"], a["id"]], a["id"])
    assert state["tags"] == [a["id"], b["id"]]
    set_entity_tags(tmp_path, "session", SID, [b["id"], a["id"]], b["id"])
    write_categories(tmp_path, [(a["id"], "A")])
    assert read_entity_tags(tmp_path, "session", SID) == {"tags": [a["id"]], "primary": a["id"]}


def test_usage_counts_ignore_orphan_session_assignments(tmp_path: Path) -> None:
    category = write_categories(tmp_path, [(None, "Review")])[0]
    existing = "aaaaaaaaaaa1"
    deleted = "aaaaaaaaaaa2"
    (tmp_path / "sessions" / existing).mkdir(parents=True)
    for sid in (existing, deleted):
        set_entity_tags(tmp_path, "session", sid, [category["id"]], category["id"])
    assert (tmp_path / "tags" / "session" / f"{deleted}.json").exists()
    assert usage_counts(tmp_path) == {category["id"]: 1}
    set_entity_tags(tmp_path, "project", "project-1", [category["id"]], category["id"])
    assert usage_counts(tmp_path, "project") == {category["id"]: 1}


@pytest.mark.parametrize("new_first", [False, True])
def test_buffered_rename_and_old_name_creation_is_order_independent(
    tmp_path: Path, new_first: bool
) -> None:
    old = write_categories(tmp_path, [(None, "Review")])[0]
    draft: list[tuple[str | None, str]] = [(old["id"], "Renamed"), (None, "Review")]
    if new_first:
        draft.reverse()
    result = write_categories(tmp_path, draft)
    by_name = {tag["name"]: tag["id"] for tag in result}
    assert by_name["Renamed"] == old["id"]
    assert by_name["Review"] != old["id"]
    assert write_categories(tmp_path, draft) == result


def test_duplicate_names_are_still_refused_with_explicit_rename(tmp_path: Path) -> None:
    old = write_categories(tmp_path, [(None, "Review")])[0]
    with pytest.raises(ValueError):
        write_categories(tmp_path, [(old["id"], "Renamed"), (None, " renamed ")])
    assert read_categories(tmp_path) == [old]


def test_the_revision_token_moves_only_on_real_mutations(tmp_path: Path) -> None:
    """The change marker is what lets the phone notice a store write cheaply.

    A token that moved on a NO-OP rewrite would be as bad as one that never
    moved: every identical save the TUI or an agent retries would repaint every
    open phone list, and the 2 s tick would do a store walk for nothing.
    """
    from local_operator.session.session_tags import read_tags_revision

    assert read_tags_revision(tmp_path) is None
    review = write_categories(tmp_path, [(None, "Review")])[0]
    first = read_tags_revision(tmp_path)
    assert isinstance(first, str) and first

    # An identical re-write lands on the same bytes: no move.
    assert write_categories(tmp_path, [(review["id"], "Review")]) == [review]
    assert read_tags_revision(tmp_path) == first

    renamed = write_categories(tmp_path, [(review["id"], "Renamed")])[0]
    second = read_tags_revision(tmp_path)
    assert second is not None and second != first

    # Assignment writes bump too, and an identical assignment does not.
    set_entity_tags(tmp_path, "session", SID, [renamed["id"]], renamed["id"])
    third = read_tags_revision(tmp_path)
    assert third is not None and third != second
    set_entity_tags(tmp_path, "session", SID, [renamed["id"]], renamed["id"])
    assert read_tags_revision(tmp_path) == third

    # Clearing is a real mutation; a repeated clear is not.
    set_entity_tags(tmp_path, "session", SID, [], None)
    fourth = read_tags_revision(tmp_path)
    assert fourth is not None and fourth != third
    set_entity_tags(tmp_path, "session", SID, [], None)
    assert read_tags_revision(tmp_path) == fourth

    # Read/assignment semantics are untouched by the marker.
    assert read_categories(tmp_path) == [renamed]
    assert read_entity_tags(tmp_path, "session", SID) == {"tags": [], "primary": None}
    assert read_categories(tmp_path) == [renamed]


@pytest.mark.parametrize(
    "contents",
    [
        "broken",
        "[]",
        '{"version":2,"token":"abc"}',
        '{"version":1,"token":123}',
        '{"version":1,"token":""}',
    ],
)
def test_missing_or_unreadable_revision_is_none(tmp_path: Path, contents: str) -> None:
    from local_operator.session.session_tags import read_tags_revision

    assert read_tags_revision(tmp_path) is None
    path = tmp_path / "tags" / "revision"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    assert read_tags_revision(tmp_path) is None
