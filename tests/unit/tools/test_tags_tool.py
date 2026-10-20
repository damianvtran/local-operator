"""The ``tags`` tool: the V2-4 agent surface over the session tag store.

These tests pin what a model reads back and, more importantly, what the store
does AFTER a refused call: names-only assignment, explicit creation, the caps,
the remote refusal and the audit-shaped receipts. The store itself is the
shipped one (``local_operator.session.session_tags``), so these are contract
tests for the tool, not a second copy of the store's own suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.resume import is_user_session
from local_operator.session.session_tags import (
    read_categories,
    read_entity_tags,
    write_categories,
)
from local_operator.tools import tags_tool
from local_operator.tools.tags_tool import build_tags_tool, execute_tags

SESSION_A = "aaaaaaaabbbb"
SESSION_B = "ccccccccdddd"
SESSION_C = "eeeeeeeeffff"


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The store root the tool resolves through ``config_dir()``."""
    monkeypatch.setattr(tags_tool, "config_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def context() -> ToolContext:
    return ToolContext(cwd=".")


def make_session(root: Path, session_id: str) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def call(context: ToolContext, **args: Any):
    return await execute_tags("tc", args, None, None, context)


async def text_of(context: ToolContext, **args: Any) -> str:
    return (await call(context, **args)).text


def payload(result: Any) -> dict[str, Any]:
    """``details`` is Optional on ToolResult; every receipt here carries one."""
    assert result.details is not None
    return result.details


def stored_names(root: Path) -> list[str]:
    return [item["name"] for item in read_categories(root)]


def test_the_tool_is_advertised_as_a_metadata_write() -> None:
    tool = build_tags_tool()
    assert tool.name == "tags"
    # Local metadata writes are the project/library precedent: no prompt.
    assert tool.approval_tier == "read"
    assert tool.concurrency == "exclusive"


@pytest.mark.asyncio
async def test_list_empty_points_at_create(root: Path, context: ToolContext) -> None:
    result = await call(context, op="list")
    assert not result.is_error
    assert "No tag categories yet" in result.text
    assert "op='create'" in result.text


@pytest.mark.asyncio
async def test_list_reports_ids_order_and_counts(root: Path, context: ToolContext) -> None:
    await text_of(context, op="create", name="Review")
    await text_of(context, op="create", name="Project")
    make_session(root, SESSION_A)
    make_session(root, SESSION_B)
    await text_of(context, op="assign", session_id=SESSION_A, tags=["Review"])
    await text_of(context, op="assign", session_id=SESSION_B, tags=["Review", "Project"])

    result = await call(context, op="list")
    rows = payload(result)["tags"]
    assert [row["order"] for row in rows] == [0, 1]
    assert [row["name"] for row in rows] == ["Review", "Project"]
    assert [row["conversations"] for row in rows] == [2, 1]
    assert all(len(row["id"]) == 10 for row in rows)
    # The receipt names the vocabulary, not just counts.
    assert "Review" in result.text and "Project" in result.text


@pytest.mark.asyncio
async def test_create_refuses_duplicate_case_blank_reserved_and_long(
    root: Path, context: ToolContext
) -> None:
    created = await text_of(context, op="create", name="Review")
    assert "Created category 'Review'" in created

    duplicate = await call(context, op="create", name="review")
    assert duplicate.is_error
    assert "already exists" in duplicate.text
    assert "Known categories: Review" in duplicate.text  # stored case preserved

    for bad in ("", "   ", "x" * 25):
        refused = await call(context, op="create", name=bad)
        assert refused.is_error, bad

    reserved = await call(context, op="create", name="Untagged")
    assert reserved.is_error and "reserved" in reserved.text

    # No silent partial: exactly the one category that was accepted exists.
    assert stored_names(root) == ["Review"]


@pytest.mark.asyncio
async def test_create_refuses_at_the_category_cap(root: Path, context: ToolContext) -> None:
    write_categories(root, [(None, f"C{i:02d}") for i in range(32)])
    refused = await call(context, op="create", name="One more")
    assert refused.is_error and "maximum 32 categories" in refused.text
    assert len(read_categories(root)) == 32


@pytest.mark.asyncio
async def test_rename_keeps_the_id_and_every_assignment(root: Path, context: ToolContext) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    review_id = read_categories(root)[0]["id"]
    await text_of(context, op="assign", session_id=SESSION_A, tags=["Review"])

    renamed = await call(context, op="rename", id=review_id, name="Code review")
    assert not renamed.is_error
    assert "assignments keep the id" in renamed.text
    assert read_categories(root) == [{"id": review_id, "name": "Code review"}]

    assignment = read_entity_tags(root, "session", SESSION_A)
    assert assignment["tags"] == [review_id]
    listed = await call(context, op="list")
    assert payload(listed)["tags"][0]["name"] == "Code review"
    assert payload(listed)["tags"][0]["conversations"] == 1


@pytest.mark.asyncio
async def test_rename_refuses_collision_unknown_id_and_missing_id(
    root: Path, context: ToolContext
) -> None:
    await text_of(context, op="create", name="Review")
    await text_of(context, op="create", name="Project")
    review_id = read_categories(root)[0]["id"]

    collision = await call(context, op="rename", id=review_id, name="project")
    assert collision.is_error and "already exists" in collision.text
    unknown = await call(context, op="rename", id="0000000000", name="Nope")
    assert unknown.is_error and "Unknown category id" in unknown.text
    missing = await call(context, op="rename", name="Nope")
    assert missing.is_error and "needs the category id" in missing.text
    assert stored_names(root) == ["Review", "Project"]


@pytest.mark.asyncio
async def test_assign_takes_names_only_and_refuses_ids(root: Path, context: ToolContext) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    review_id = read_categories(root)[0]["id"]

    refused = await call(context, op="assign", session_id=SESSION_A, tags=[review_id])
    assert refused.is_error
    assert "Unknown tags" in refused.text
    assert read_entity_tags(root, "session", SESSION_A)["tags"] == []


@pytest.mark.asyncio
async def test_assign_refuses_a_misspelling_with_the_known_vocabulary(
    root: Path, context: ToolContext
) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    refused = await call(context, op="assign", session_id=SESSION_A, tags=["reveiw"])
    assert refused.is_error
    assert "reveiw" in refused.text and "Known categories: Review" in refused.text
    # Never auto-created from a failed spelling.
    assert stored_names(root) == ["Review"]


@pytest.mark.asyncio
async def test_assign_is_case_insensitive_but_returns_stored_names(
    root: Path, context: ToolContext
) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Code Review")
    result = await call(context, op="assign", session_id=SESSION_A, tags=["code review"])
    assert not result.is_error
    assert payload(result)["tags"] == ["Code Review"]
    assert "Code Review" in result.text


@pytest.mark.asyncio
async def test_assign_refuses_duplicates_and_more_than_eight(
    root: Path, context: ToolContext
) -> None:
    make_session(root, SESSION_A)
    write_categories(root, [(None, f"C{i}") for i in range(9)])

    duplicated = await call(context, op="assign", session_id=SESSION_A, tags=["C0", "c0"])
    assert duplicated.is_error and "Duplicate tags" in duplicated.text

    oversized = await call(
        context, op="assign", session_id=SESSION_A, tags=[f"C{i}" for i in range(9)]
    )
    assert oversized.is_error and "1..8" in oversized.text
    assert read_entity_tags(root, "session", SESSION_A)["tags"] == []


@pytest.mark.asyncio
async def test_assign_at_the_eight_tag_cap_lands(root: Path, context: ToolContext) -> None:
    make_session(root, SESSION_A)
    write_categories(root, [(None, f"C{i}") for i in range(8)])
    result = await call(
        context, op="assign", session_id=SESSION_A, tags=[f"C{i}" for i in range(8)]
    )
    assert not result.is_error
    assert len(read_entity_tags(root, "session", SESSION_A)["tags"]) == 8


@pytest.mark.asyncio
async def test_assign_empty_list_clears(root: Path, context: ToolContext) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    await text_of(context, op="assign", session_id=SESSION_A, tags=["Review"])

    cleared = await call(context, op="assign", session_id=SESSION_A, tags=[])
    assert not cleared.is_error
    assert "Cleared tags on session" in cleared.text
    assert payload(cleared)["tags"] == [] and payload(cleared)["primary"] is None
    assert read_entity_tags(root, "session", SESSION_A)["tags"] == []


@pytest.mark.asyncio
async def test_assign_refuses_primary_outside_the_assigned_names(
    root: Path, context: ToolContext
) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    await text_of(context, op="create", name="Project")

    refused = await call(
        context, op="assign", session_id=SESSION_A, tags=["Review"], primary="Project"
    )
    assert refused.is_error and "must be one of the assigned names" in refused.text
    assert read_entity_tags(root, "session", SESSION_A)["tags"] == []

    empty_primary = await call(
        context, op="assign", session_id=SESSION_A, tags=[], primary="Review"
    )
    assert empty_primary.is_error


@pytest.mark.asyncio
async def test_omitting_primary_retains_a_second_position_primary(
    root: Path, context: ToolContext
) -> None:
    """The operator's chosen primary is not assignment order (R-1).

    The store's own ``primary=None`` fallback is ``ordered[0]`` — the oldest
    surviving assignment. A second-position primary therefore disappears
    unless the tool reads the current assignment and passes it through, which
    is what keeps "adding another tag never moves it" true for agent writes.
    """
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    await text_of(context, op="create", name="Project")
    project_id = {item["name"]: item["id"] for item in read_categories(root)}["Project"]
    await text_of(
        context,
        op="assign",
        session_id=SESSION_A,
        tags=["Review", "Project"],
        primary="Project",
    )

    kept = await call(context, op="assign", session_id=SESSION_A, tags=["Project", "Review"])
    assert not kept.is_error
    assert read_entity_tags(root, "session", SESSION_A)["primary"] == project_id
    assert payload(kept)["primary"] == "Project"

    # When the primary does NOT survive, the store's first-in-order fallback is
    # correct and the receipt reports the promoted survivor.
    dropped = await call(context, op="assign", session_id=SESSION_A, tags=["Review"])
    assert not dropped.is_error
    assert payload(dropped)["primary"] == "Review"


@pytest.mark.asyncio
async def test_a_failed_assign_leaves_the_previous_assignment_untouched(
    root: Path, context: ToolContext
) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    await text_of(context, op="assign", session_id=SESSION_A, tags=["Review"])
    before = read_entity_tags(root, "session", SESSION_A)

    refused = await call(context, op="assign", session_id=SESSION_A, tags=["Review", "missing"])
    assert refused.is_error
    assert read_entity_tags(root, "session", SESSION_A) == before


@pytest.mark.asyncio
async def test_delegated_and_hidden_sessions_are_taggable(root: Path, context: ToolContext) -> None:
    directory = make_session(root, SESSION_B)
    (directory / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    assert is_user_session(directory) is False  # hidden from human lists

    await text_of(context, op="create", name="Review")
    result = await call(context, op="assign", session_id=SESSION_B, tags=["Review"])
    assert not result.is_error
    assert SESSION_B in result.text
    assert read_entity_tags(root, "session", SESSION_B)["tags"] != []


@pytest.mark.asyncio
async def test_a_remote_session_is_refused_with_the_route_code(
    root: Path, context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.session.remote_open as remote_open

    await text_of(context, op="create", name="Review")
    monkeypatch.setattr(
        remote_open,
        "remote_row_for",
        lambda session_id, store_root: SimpleNamespace(
            owner_device="dev-1", owner_device_name="Studio Mac"
        ),
    )
    refused = await call(context, op="assign", session_id=SESSION_C, tags=["Review"])
    assert refused.is_error
    assert "session_is_remote" in refused.text and "Studio Mac" in refused.text
    assert payload(refused)["code"] == "session_is_remote"
    # Nothing was written for the peer's id.
    assert not (root / "tags" / "session" / f"{SESSION_C}.json").exists()


@pytest.mark.asyncio
async def test_an_unknown_session_is_refused(
    root: Path, context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.session.remote_open as remote_open

    await text_of(context, op="create", name="Review")
    monkeypatch.setattr(remote_open, "remote_row_for", lambda *_args: None)
    refused = await call(context, op="assign", session_id=SESSION_C, tags=["Review"])
    assert refused.is_error and "Unknown session" in refused.text


@pytest.mark.asyncio
async def test_list_counts_exclude_deleted_directories_but_keep_hidden_ones(
    root: Path, context: ToolContext
) -> None:
    await text_of(context, op="create", name="Review")
    make_session(root, SESSION_A)
    hidden = make_session(root, SESSION_B)
    (hidden / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    deleted = make_session(root, SESSION_C)

    for session_id in (SESSION_A, SESSION_B, SESSION_C):
        await text_of(context, op="assign", session_id=session_id, tags=["Review"])
    deleted.rmdir()

    listed = await call(context, op="list")
    assert payload(listed)["tags"][0]["conversations"] == 2


@pytest.mark.asyncio
async def test_receipts_always_name_the_session_and_the_categories(
    root: Path, context: ToolContext
) -> None:
    make_session(root, SESSION_A)
    await text_of(context, op="create", name="Review")
    receipt = await call(context, op="assign", session_id=SESSION_A, tags=["Review"])
    assert SESSION_A in receipt.text and "Review" in receipt.text
    assert payload(receipt)["session_id"] == SESSION_A
    assert payload(receipt)["tags"] == ["Review"]
    assert payload(receipt)["primary"] == "Review"
    assert "tags_revision" in payload(receipt)
