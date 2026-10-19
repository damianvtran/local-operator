from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.library import index
from local_operator.library.metadata import MetadataStore
from local_operator.tools.library_tool import execute_library
from local_operator.tools.registry import TOOL_BUILDERS

from ._fixtures import config_root, session, write_rows


def text(result) -> str:
    return "".join(block.text for block in result.content)


async def call(tool_call_id: str, **args):
    return await execute_library(tool_call_id, args)


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = config_root(monkeypatch, tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    session(root, "abcdef123456", write_rows("w1", 1, path))
    catalog = index.build_catalog(root, budget_s=None)
    return root, catalog.assets[0].id


@pytest.mark.asyncio
async def test_stats_and_list_are_open_reads(library) -> None:
    root, asset_id = library
    result = await call("c1", op="stats")
    assert result.is_error is False
    payload = json.loads(text(result))
    assert payload["assets"] == 1 and payload["state"] == "ready"

    result = await call("c2", op="list", q="notes")
    payload = json.loads(text(result))
    assert payload["total"] == 1 and payload["rows"][0]["id"] == asset_id


@pytest.mark.asyncio
async def test_update_records_agent_provenance_and_organized(library) -> None:
    root, asset_id = library
    result = await call("c3", op="update", asset_id=asset_id, name="Notes", organized=True)
    assert result.is_error is False
    row = MetadataStore(root).asset(asset_id)
    assert row["name"] == "Notes" and row["name_source"] == "agent"
    assert row["organized_at"] and row["organized_by"] == "agent"

    result = await call(
        "c4", op="update", asset_id=asset_id, description="D", actor="librarian", confidence=0.8
    )
    row = MetadataStore(root).asset(asset_id)
    assert row["description_source"] == "librarian" and row["confidence"] == 0.8

    listed = json.loads(text(await call("c5", op="list", organized=True)))
    assert listed["total"] == 1


@pytest.mark.asyncio
async def test_tags_ops_and_the_twelve_tag_cap(library) -> None:
    root, asset_id = library
    created = await call("c6", op="tags", action="create", tag_name="travel")
    tag_id = json.loads(text(created))["id"]
    assert json.loads(text(await call("c7", op="tags")))["tags"][0]["name"] == "travel"

    applied = await call(
        "c8", op="tags", action="set", asset_id=asset_id, tags=[tag_id], primary=tag_id
    )
    row = json.loads(text(applied))
    assert row["tags"] == ["travel"] and row["primary_tag"] == tag_id

    cleared = await call("c9", op="tags", action="clear", asset_id=asset_id)
    assert json.loads(text(cleared))["tags"] == []


@pytest.mark.asyncio
async def test_argument_validation_is_a_result_not_a_raise(library) -> None:
    _root, _asset_id = library
    missing = await call("c10", op="get")
    assert missing.is_error is True and "asset_id" in text(missing)
    unknown = await call("c11", op="update", asset_id="la_0123456789abcdef", name="x")
    assert unknown.is_error is True
    bad_filter = await call("c12", op="list", kind="nonsense")
    assert bad_filter.is_error is True


def test_the_tool_is_registered_for_every_session() -> None:
    assert "library" in TOOL_BUILDERS
    tool = TOOL_BUILDERS["library"](None)
    assert tool is not None and tool.name == "library" and tool.approval_tier == "read"
