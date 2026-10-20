"""The durable registry and assignment contract, without a server/runtime."""

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.session_tags import (
    TagCreateApplyError,
    effective_primary,
    read_categories,
    read_entity_tags,
    read_tags_revision,
    resolve_category_names,
    set_entity_tags,
    toggle_entity_tag,
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
    assert read_entity_tags(tmp_path, "session", SID) == {
        "tags": [b["id"]],
        "primary": b["id"],
        "revision": 1,
    }
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
    empty = {"tags": [], "primary": None, "revision": 0}
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
        "revision": 0,
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
    assert state == {"tags": [a["id"], b["id"]], "primary": b["id"], "revision": 2}
    state = set_entity_tags(tmp_path, "session", SID, [b["id"], a["id"]], a["id"])
    assert state["tags"] == [a["id"], b["id"]]
    set_entity_tags(tmp_path, "session", SID, [b["id"], a["id"]], b["id"])
    write_categories(tmp_path, [(a["id"], "A")])
    assert read_entity_tags(tmp_path, "session", SID) == {
        "tags": [a["id"]],
        "primary": a["id"],
        "revision": 4,
    }


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
    assert read_entity_tags(tmp_path, "session", SID) == {
        "tags": [],
        "primary": None,
        "revision": 2,
    }
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
    assert read_tags_revision(tmp_path) is None
    path = tmp_path / "tags" / "revision"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    assert read_tags_revision(tmp_path) is None


def test_a_failed_marker_write_removes_the_stale_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A landed assignment must stay observable when its marker cannot be written.

    The window is the one review round 1 MAJOR-1 reproduced: the assignment
    replace lands, then the marker write raises. Raising alone leaves an
    old-but-valid token, and the client's identical retry writes no bytes
    (``_write`` returns False), so nothing ever bumps it. The recovery is
    removing the stale marker, which makes readers use the stats fingerprint
    the landed assignment has already moved.
    """
    from local_operator.session import session_tags

    (tmp_path / "sessions" / SID).mkdir(parents=True)
    review = write_categories(tmp_path, [(None, "Review")])[0]
    marker = tmp_path / "tags" / "revision"
    assert read_tags_revision(tmp_path) is not None

    real_write = session_tags._write
    failures = {"left": 2}

    def flaky(path: Path, value: dict[str, Any]) -> bool:
        if path == marker and failures["left"]:
            failures["left"] -= 1
            raise OSError("injected marker write failure")
        return real_write(path, value)

    monkeypatch.setattr(session_tags, "_write", flaky)

    # Both attempts fail; the assignment write itself still lands.
    assert set_entity_tags(tmp_path, "session", SID, [review["id"]], review["id"]) == {
        "tags": [review["id"]],
        "primary": review["id"],
        "revision": 1,
    }
    assert failures["left"] == 0, "the retry must actually be attempted"
    assert not marker.exists(), "the stale marker must be removed"
    assert read_tags_revision(tmp_path) is None

    # Not one-way: the next real write recreates the marker.
    monkeypatch.setattr(session_tags, "_write", real_write)
    set_entity_tags(tmp_path, "session", SID, [], None)
    assert read_tags_revision(tmp_path) is not None


def test_assignment_revisions_track_changes_not_retries(tmp_path: Path) -> None:
    a, b = [tag["id"] for tag in write_categories(tmp_path, [(None, "A"), (None, "B")])]
    path = tmp_path / "tags" / "session" / f"{SID}.json"
    assert read_entity_tags(tmp_path, "session", SID)["revision"] == 0
    assert set_entity_tags(tmp_path, "session", SID, [], None)["revision"] == 0
    assert not path.exists()
    for tags, primary, expected, revision in [
        ([a], a, "0", 1),
        ([a, b], a, 1, 2),
        ([a, b], b, "2", 3),
        ([b], b, 3, 4),
        ([], None, "4", 5),
    ]:
        assert (
            set_entity_tags(tmp_path, "session", SID, tags, primary, expected)["revision"]
            == revision
        )
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        assert (
            set_entity_tags(tmp_path, "session", SID, tags, primary, str(revision))["revision"]
            == revision
        )
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    set_entity_tags(tmp_path, "session", SID, [a, b], a)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert set_entity_tags(tmp_path, "session", SID, [b, a], None, "6")["revision"] == 6
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_revision_conflict_is_typed_and_does_not_write(tmp_path: Path) -> None:
    from local_operator.session.session_tags import TagRevisionConflict

    tag = write_categories(tmp_path, [(None, "A")])[0]
    set_entity_tags(tmp_path, "session", SID, [tag["id"]], tag["id"])
    path = tmp_path / "tags" / "session" / f"{SID}.json"
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    for expected in (0, "0", "01"):
        with pytest.raises(TagRevisionConflict) as caught:
            set_entity_tags(tmp_path, "session", SID, [], None, expected)
        assert caught.value.current == {"tags": [tag], "primary": tag["id"], "tags_revision": "1"}
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_conditional_writers_serialize_within_one_process(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from local_operator.session.session_tags import TagRevisionConflict

    a, b = [tag["id"] for tag in write_categories(tmp_path, [(None, "A"), (None, "B")])]
    barrier = Barrier(2)

    def apply(key):
        barrier.wait(timeout=5)
        try:
            return set_entity_tags(tmp_path, "session", SID, [key], key, "0")["revision"]
        except TagRevisionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(apply, [a, b]))
    assert sorted(map(str, results)) == ["1", "conflict"]
    assert read_entity_tags(tmp_path, "session", SID)["revision"] == 1


def test_legacy_assignment_starts_at_zero_without_rewriting_on_retry(tmp_path: Path) -> None:
    tag = write_categories(tmp_path, [(None, "A")])[0]["id"]
    path = tmp_path / "tags" / "session" / f"{SID}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "tags": [tag], "primary": tag}))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert read_entity_tags(tmp_path, "session", SID)["revision"] == 0
    assert set_entity_tags(tmp_path, "session", SID, [tag], tag, "0")["revision"] == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert set_entity_tags(tmp_path, "session", SID, [], None, "0")["revision"] == 1


# -- one toggle/create-and-apply semantic -------------------------------------


def test_toggle_creates_and_applies_a_new_name_once(tmp_path: Path) -> None:
    """An exact new name is ONE activation: create the category, assign it, primary."""
    first = toggle_entity_tag(tmp_path, "session", SID, "Deep Work")
    assert first["created"] is True
    assert first["applied"] is True
    assert read_categories(tmp_path) == [{"id": first["primary"], "name": "Deep Work"}]
    assert read_entity_tags(tmp_path, "session", SID) == {
        "tags": [first["primary"]],
        "primary": first["primary"],
        "revision": 1,
    }


def test_toggle_uses_existing_name_case_insensitively(tmp_path: Path) -> None:
    tag = write_categories(tmp_path, [(None, "Review")])[0]["id"]
    result = toggle_entity_tag(tmp_path, "session", SID, " review ")
    assert result == {
        "tags": [tag],
        "primary": tag,
        "revision": 1,
        "created": False,
        "applied": True,
        "name": "Review",
    }
    assert read_categories(tmp_path) == [{"id": tag, "name": "Review"}]


def test_toggle_primary_rules_across_add_and_remove(tmp_path: Path) -> None:
    """First assigned is primary; adding never moves it; removing it promotes oldest."""
    a = toggle_entity_tag(tmp_path, "session", SID, "A")
    a_id = a["tags"][0]
    b = toggle_entity_tag(tmp_path, "session", SID, "B")
    b_id = b["tags"][-1]
    c = toggle_entity_tag(tmp_path, "session", SID, "C")
    c_id = c["tags"][-1]
    state = read_entity_tags(tmp_path, "session", SID)
    assert state["tags"] == [a_id, b_id, c_id]
    assert state["primary"] == a_id

    removed_b = toggle_entity_tag(tmp_path, "session", SID, "B")
    assert removed_b["applied"] is False
    assert removed_b["tags"] == [a_id, c_id]
    assert removed_b["primary"] == a_id

    removed_a = toggle_entity_tag(tmp_path, "session", SID, "A")
    assert removed_a["tags"] == [c_id]
    assert removed_a["primary"] == c_id

    emptied = toggle_entity_tag(tmp_path, "session", SID, "C")
    assert emptied["tags"] == []
    assert emptied["primary"] is None


def test_toggle_checks_assignment_cap_before_creating(tmp_path: Path) -> None:
    for index in range(8):
        toggle_entity_tag(tmp_path, "session", SID, f"T{index}")
    registry = read_categories(tmp_path)
    with pytest.raises(ValueError, match="At most 8"):
        toggle_entity_tag(tmp_path, "session", SID, "Too Many")
    assert read_categories(tmp_path) == registry


def test_toggle_checks_registry_cap_before_creating(tmp_path: Path) -> None:
    write_categories(tmp_path, [(None, str(index)) for index in range(32)])
    with pytest.raises(ValueError, match="At most 32"):
        toggle_entity_tag(tmp_path, "session", SID, "One More")
    assert read_entity_tags(tmp_path, "session", SID)["tags"] == []


def test_create_then_assignment_failure_reports_which_half_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two-write seam is disclosed, never reported as an atomic failure.

    Retrying the CREATE would duplicate the name; retrying the APPLY is right.
    The exception carries that distinction so the receipt can say so.
    """

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("local_operator.session.session_tags._set_entity_tags", fail)
    with pytest.raises(TagCreateApplyError) as caught:
        toggle_entity_tag(tmp_path, "session", SID, "New")
    assert caught.value.created is True
    assert [item["name"] for item in read_categories(tmp_path)] == ["New"]


def test_effective_primary_requires_registered_and_assigned(tmp_path: Path) -> None:
    a = write_categories(tmp_path, [(None, "A")])[0]["id"]
    b = write_categories(tmp_path, [(a, "A"), (None, "B")])[1]["id"]
    categories = read_categories(tmp_path)
    assert effective_primary([a], a, categories) == a
    # not assigned: the id is valid but this entity is Untagged for grouping
    assert effective_primary([b], a, categories) is None
    # removed from the registry: a dangling assignment falls to Untagged
    assert effective_primary([a], a, []) is None
    assert effective_primary([], None, categories) is None
