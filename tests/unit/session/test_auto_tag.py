"""The fork auto-tag sidecar: record, read, spend, and the conditional write."""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.session import auto_tag, session_tags

SID = "abcdef123456"
OTHER = "abcdef123457"


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    (root / "sessions" / SID).mkdir(parents=True)
    (root / "sessions" / OTHER).mkdir(parents=True)
    return root


def _categories(root: Path) -> list[dict[str, str]]:
    return session_tags.write_categories(root, [(None, "Review"), (None, "Project")])


def test_record_fork_snapshots_the_childs_current_revision(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _categories(root)

    assert auto_tag.record_fork(root, SID) is True
    raw = json.loads((root / "sessions" / SID / auto_tag.AUTO_TAG_FILENAME).read_text())
    assert raw == {"version": 1, "auto_tag": True, "tags_revision": "0"}
    intent = auto_tag.read(root, SID)
    assert intent is not None
    assert (intent.root, intent.session_id, intent.revision) == (root, SID, "0")


def test_record_fork_skips_a_child_that_already_carries_tags(tmp_path: Path) -> None:
    root = _root(tmp_path)
    review, _ = _categories(root)
    session_tags.set_entity_tags(root, "session", SID, [review["id"]], review["id"])

    assert auto_tag.record_fork(root, SID) is False
    assert not (root / "sessions" / SID / auto_tag.AUTO_TAG_FILENAME).exists()


def test_read_is_off_for_a_missing_malformed_or_disabled_sidecar(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert auto_tag.read(root, SID) is None

    sidecar = root / "sessions" / SID / auto_tag.AUTO_TAG_FILENAME
    sidecar.write_text("{not json")
    assert auto_tag.read(root, SID) is None

    sidecar.write_text(json.dumps({"version": 1, "auto_tag": False, "tags_revision": "0"}))
    assert auto_tag.read(root, SID) is None

    sidecar.write_text(json.dumps({"version": 1, "auto_tag": True}))
    assert auto_tag.read(root, SID) is None


def test_settle_writes_once_with_the_recorded_revision(tmp_path: Path) -> None:
    root = _root(tmp_path)
    review, project = _categories(root)
    assert auto_tag.record_fork(root, SID)
    intent = auto_tag.read(root, SID)
    assert intent is not None

    state = auto_tag.settle(intent, [review["id"], project["id"]])
    assert state is not None
    assert state["tags"] == [review["id"], project["id"]]
    assert state["primary"] == review["id"]
    assert not (root / "sessions" / SID / auto_tag.AUTO_TAG_FILENAME).exists()

    # The revision moved, so the same intent cannot write a second time.
    assert auto_tag.settle(intent, [project["id"]]) is None
    current = session_tags.read_entity_tags(root, "session", SID)
    assert current["tags"] == [review["id"], project["id"]]


def test_settle_loses_to_a_manual_add_then_clear_before_the_first_turn(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    review, _ = _categories(root)
    assert auto_tag.record_fork(root, SID)
    intent = auto_tag.read(root, SID)
    assert intent is not None

    # The human assigns and then clears before the first substantive turn. The
    # final VALUE is empty, but the revision moved twice; eligibility was the
    # snapshot, so auto-tagging must skip.
    session_tags.set_entity_tags(root, "session", SID, [review["id"]], review["id"])
    session_tags.set_entity_tags(root, "session", SID, [], None)

    assert auto_tag.settle(intent, [review["id"]]) is None
    assert session_tags.read_entity_tags(root, "session", SID)["tags"] == []
    assert not (root / "sessions" / SID / auto_tag.AUTO_TAG_FILENAME).exists()


def test_settle_drops_a_category_deleted_during_the_provider_call(tmp_path: Path) -> None:
    root = _root(tmp_path)
    review, project = _categories(root)
    assert auto_tag.record_fork(root, SID)
    intent = auto_tag.read(root, SID)
    assert intent is not None

    session_tags.write_categories(root, [(project["id"], project["name"])])
    assert auto_tag.settle(intent, [review["id"]]) is None
    assert session_tags.read_entity_tags(root, "session", SID)["tags"] == []

    # ...and a surviving id still lands while the deleted one is dropped.
    assert auto_tag.record_fork(root, OTHER)
    second = auto_tag.read(root, OTHER)
    assert second is not None
    state = auto_tag.settle(second, [review["id"], project["id"]])
    assert state is not None
    assert state["tags"] == [project["id"]]


def test_settle_never_creates_a_category_from_model_output(tmp_path: Path) -> None:
    root = _root(tmp_path)
    review, project = _categories(root)
    assert auto_tag.record_fork(root, SID)
    intent = auto_tag.read(root, SID)
    assert intent is not None

    assert auto_tag.settle(intent, ["deadbeef00"]) is None
    assert [item["id"] for item in session_tags.read_categories(root)] == [
        review["id"],
        project["id"],
    ]


def test_settle_spends_the_intent_even_when_it_writes_nothing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _categories(root)
    assert auto_tag.record_fork(root, SID)
    intent = auto_tag.read(root, SID)
    assert intent is not None

    assert auto_tag.settle(intent, []) is None
    assert not (root / "sessions" / SID / auto_tag.AUTO_TAG_FILENAME).exists()
