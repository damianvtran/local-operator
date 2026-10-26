from __future__ import annotations

import json

import pytest

from local_operator.harness.types import ToolContext
from local_operator.librarian.state import write_state
from local_operator.library import indexer, service
from local_operator.library.metadata import MetadataStore
from local_operator.library.store import LibraryStore
from local_operator.tools.library_tool import build_library_tool, execute_library
from local_operator.tools.registry import TOOL_BUILDERS

from ._fixtures import OTHER, SESSION, config_root, session, write_rows

#: The session id ``librarian/state.json`` names in the tests below. Her id is
#: the WHOLE identity check, so a test that wants her rank writes this file.
LIBRARIAN = "f3cce3edac6a"


def text(result) -> str:
    return "".join(block.text for block in result.content)


async def call(tool_call_id: str, *, context: ToolContext | None = None, **args):
    return await execute_library(tool_call_id, args, None, None, context)


def is_librarian(root) -> None:
    """Name :data:`LIBRARIAN` as her session, through the real state writer."""
    write_state(root, {"session_id": LIBRARIAN})


def librarian_context(session_id: str = LIBRARIAN) -> ToolContext:
    return ToolContext(session_id=session_id)


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = config_root(monkeypatch, tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    session(root, "abcdef123456", write_rows("w1", 1, path))
    indexer.rebuild_now(root)
    asset_id = LibraryStore(root).query_page(service.Query(), limit=1)[0][0].id
    return root, asset_id


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
    assert row is not None
    assert row["name"] == "Notes" and row["name_source"] == "agent"
    assert row["organized_at"] and row["organized_by"] == "agent"

    # Her rank is now DERIVED from the caller's session, so the explicit
    # ``actor`` below only works because this session IS hers.
    is_librarian(root)
    result = await call(
        "c4",
        context=librarian_context(),
        op="update",
        asset_id=asset_id,
        description="D",
        actor="librarian",
        confidence=0.8,
    )
    row = MetadataStore(root).asset(asset_id)
    assert row is not None
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
    tool = build_library_tool()
    assert tool.name == "library" and tool.approval_tier == "read"


def test_the_wire_schema_matches_the_validator() -> None:
    from local_operator.tools.library_tool import LIBRARY_PARAMETERS, LibraryParams

    properties = LIBRARY_PARAMETERS["properties"]
    assert set(properties) == set(LibraryParams.model_fields)
    assert LIBRARY_PARAMETERS["required"] == ["op"]
    assert properties["op"]["enum"] == [
        "stats",
        "list",
        "get",
        "update",
        "tags",
        "conventions",
    ]
    # A compact wire schema is a second declaration; the property set above is
    # the guard that keeps it from drifting from the model that validates calls.


# -- who the caller is decides the provenance (the organized=0 bug) ----------


@pytest.fixture
def two_assets(tmp_path, monkeypatch):
    """Two real catalog assets and NO Librarian, for the identity cases."""
    root = config_root(monkeypatch, tmp_path)
    first = tmp_path / "notes.txt"
    first.write_text("hi")
    second = tmp_path / "plan.txt"
    second.write_text("hi")
    session(root, SESSION, write_rows("w1", 1, first) + write_rows("w2", 2, second))
    indexer.rebuild_now(root)
    store = LibraryStore(root)
    ids = {
        "notes": store.query_page(service.Query(q="notes"), limit=1)[0][0].id,
        "plan": store.query_page(service.Query(q="plan"), limit=1)[0][0].id,
    }
    assert ids["notes"] != ids["plan"]
    return root, ids


@pytest.mark.asyncio
async def test_the_librarian_is_derived_from_her_session_and_organizes(two_assets) -> None:
    """(i) The real entry point, her session, no ``actor`` argument at all.

    The bug this pins: the tool's ``actor`` defaulted to ``agent`` and the seed
    never passed one, so four days of her naming and tagging was recorded as an
    ordinary agent's and ``organized_at`` was never stamped.
    """
    root, ids = two_assets
    is_librarian(root)
    result = await call(
        "c20",
        context=librarian_context(),
        op="update",
        asset_id=ids["notes"],
        name="Operator notes",
        description="What the notes are.",
    )
    assert result.is_error is False, text(result)
    row = MetadataStore(root).asset(ids["notes"])
    assert row is not None
    assert row["name_source"] == "librarian" and row["description_source"] == "librarian"
    assert row["organized_by"] == "librarian" and row["organized_at"]

    # A tag set ALONE is organizing work: applied metadata, not a favorite.
    created = await call(
        "c21", context=librarian_context(), op="tags", action="create", tag_name="travel"
    )
    tag_id = json.loads(text(created))["id"]
    applied = await call(
        "c22",
        context=librarian_context(),
        op="tags",
        action="set",
        asset_id=ids["plan"],
        tags=[tag_id],
    )
    assert applied.is_error is False, text(applied)
    tagged = MetadataStore(root).asset(ids["plan"])
    assert tagged is not None
    assert tagged["organized_by"] == "librarian" and tagged["organized_at"]
    assert tagged["name_source"] == "derived", "organizing must not claim a name she never wrote"
    # The tag's own confidence column is where confidence belongs; the asset
    # keeps none, because a tag set says nothing about the asset's.
    assert tagged["confidence"] is None


@pytest.mark.asyncio
async def test_a_non_librarian_session_records_agent_and_does_not_organize(library) -> None:
    """(ii) Every other session on the machine: agent, and no organization."""
    root, asset_id = library
    is_librarian(root)
    result = await call(
        "c23", context=ToolContext(session_id=OTHER), op="update", asset_id=asset_id, name="Mine"
    )
    assert result.is_error is False, text(result)
    row = MetadataStore(root).asset(asset_id)
    assert row is not None
    assert row["name_source"] == "agent"
    assert row["organized_at"] is None and row["organized_by"] is None


@pytest.mark.asyncio
async def test_a_child_of_the_librarian_is_not_the_librarian(library) -> None:
    """(v) Her SUBAGENT is a different session, so it writes as an agent.

    ``harness/subagent.py`` gives the child a fresh ``sessions/<id>`` directory
    and sets its context's ``session_id`` from that directory's name, inheriting
    only the parent's ``agent_id``. The derivation reads ``session_id``, so a
    child of hers that decides to write to the catalog is recorded as an
    ordinary agent and organizes nothing — and it cannot claim her rank either,
    which is where the child used to be honoured.
    """
    root, asset_id = library
    is_librarian(root)
    child = ToolContext(
        session_id="9f2b1c0d4e5a6071",
        agent_id="librarian-agent",
        job_id="job_child",
    )
    result = await call(
        "c27", context=child, op="update", asset_id=asset_id, name="A child's title"
    )
    assert result.is_error is False, text(result)
    row = MetadataStore(root).asset(asset_id)
    assert row is not None
    assert row["name_source"] == "agent" and row["organized_at"] is None

    claimed = await call(
        "c28",
        context=child,
        op="update",
        asset_id=asset_id,
        name="Claimed",
        actor="librarian",
    )
    assert claimed.is_error is True
    assert "reserved for the Librarian session" in text(claimed)
    assert MetadataStore(root).asset(asset_id) == row


@pytest.mark.asyncio
async def test_claiming_to_be_the_librarian_is_refused(library) -> None:
    """(iii) ``actor='librarian'`` is reserved: an error, and nothing written."""
    root, asset_id = library
    is_librarian(root)
    store = MetadataStore(root)
    before = store.asset(asset_id)
    edits_before = len(store.edits(asset_id, limit=500))

    result = await call(
        "c24",
        context=ToolContext(session_id=OTHER),
        op="update",
        asset_id=asset_id,
        name="Claimed",
        actor="librarian",
    )
    assert result.is_error is True
    assert "reserved for the Librarian session" in text(result)
    after = store.asset(asset_id)
    assert after == before
    assert len(store.edits(asset_id, limit=500)) == edits_before


@pytest.mark.asyncio
async def test_the_librarian_can_still_record_an_agent_write(library) -> None:
    """(iv) Her session, an explicit ``actor='agent'``: recorded as an agent's.

    The derivation is a DEFAULT, not a promotion: a call that says what it is
    keeps saying it.
    """
    root, asset_id = library
    is_librarian(root)
    result = await call(
        "c25",
        context=librarian_context(),
        op="update",
        asset_id=asset_id,
        name="Mine",
        actor="agent",
    )
    assert result.is_error is False, text(result)
    row = MetadataStore(root).asset(asset_id)
    assert row is not None
    assert row["name_source"] == "agent" and row["organized_at"] is None


@pytest.mark.asyncio
async def test_an_install_without_her_never_records_librarian(two_assets) -> None:
    """No ``state.json`` at all is the ordinary install: agent, always."""
    root, ids = two_assets
    result = await call(
        "c26", context=librarian_context(), op="update", asset_id=ids["notes"], name="Mine"
    )
    assert result.is_error is False, text(result)
    row = MetadataStore(root).asset(ids["notes"])
    assert row is not None
    assert row["name_source"] == "agent" and row["organized_at"] is None
