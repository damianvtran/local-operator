"""Cross-package contracts: the settings registry and the backend's tool.

The ``library`` tool lives in ``local_operator/tools/library_tool.py`` (owned
by the backend-core slice). This file drives the REAL tool from a real
Librarian session, so a change to its ops or its provenance rules fails here
rather than in an unattended pass.
"""

from __future__ import annotations

import json

import pytest

from local_operator import settings_io
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.librarian import bootstrap, proactive
from local_operator.library import index, indexer
from local_operator.library.metadata import MetadataStore
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.library_tool import build_library_tool


def test_the_settings_registry_matches_the_consumers_defaults() -> None:
    """The registry's literals must equal the constants the engine reads.

    ``settings_io`` keeps its defaults as literals so it never imports the
    feature package on every CLI start; this is the pin that keeps that split
    from drifting.
    """
    rows = {setting.key: setting for setting in settings_io.SETTINGS}
    assert rows["librarian.enabled"].default is proactive.DEFAULT_ENABLED
    assert rows["librarian.name"].default == bootstrap.SESSION_TITLE
    assert rows["librarian.cadence.at"].default == proactive.DEFAULT_CADENCE_AT
    assert rows["librarian.cadence.paused"].default is proactive.DEFAULT_PAUSED
    assert (
        rows["librarian.cadence.max_extra_per_day"].default == proactive.DEFAULT_MAX_EXTRA_PER_DAY
    )
    assert rows["librarian.cadence.min_gap_minutes"].default == proactive.DEFAULT_MIN_GAP_MINUTES
    # The name cap is shared between the facade and the display reader.
    assert settings_io._LIBRARIAN_NAME_MAX_CHARS == bootstrap.MAX_NAME_CHARS


def test_the_kill_switch_is_documented_in_the_package() -> None:
    assert bootstrap.state.ENV_DISABLE == "LOCAL_OPERATOR_NO_LIBRARIAN"


@pytest.mark.asyncio
async def test_the_librarian_drives_the_real_library_tool(isolated_root, tmp_path) -> None:
    """The seed's whole job, exercised through the tool it names.

    A real Librarian session exists (the bootstrap's), a real journal seeds one
    asset, and every call below is the same ``execute_library`` entry point the
    runtime registers. The two protections the seed promises are asserted at
    the store, not in prose: a USER field survives a Librarian write, and a
    locked row refuses her.
    """
    from local_operator.librarian import bootstrap
    from local_operator.library import index
    from local_operator.library.metadata import MetadataStore
    from local_operator.tools.library_tool import execute_library
    from tests.unit.library._fixtures import session as seed_session
    from tests.unit.library._fixtures import write_rows

    def text(result) -> str:
        return "".join(block.text for block in result.content)

    sid = await bootstrap.ensure_session(isolated_root)
    assert sid
    # Every write below passes HER OWN session id: the tool derives the actor
    # from the caller's identity, so a call with no context is an ordinary
    # agent's no matter what ``actor`` says.
    hers = ToolContext(session_id=sid)
    report = tmp_path / "weekly.md"
    report.write_text("the Monday numbers", encoding="utf-8")
    seed_session(isolated_root, "feedfacecafe", write_rows("w1", 1.0, report))
    asset = index.build_catalog(isolated_root, budget_s=None).assets[0]
    # The tool reads the derived index, which the background worker owns. A
    # first look on a fresh config is legitimately `building` with no rows (the
    # service starts the worker and answers immediately), so this test builds
    # the index explicitly the way the other library fixtures do.
    from local_operator.library import indexer

    indexer.reset_for_tests()
    indexer.rebuild_now(isolated_root)

    stats = json.loads(text(await execute_library("c1", {"op": "stats"}, None, None, None)))
    assert stats["assets"] == 1 and stats["state"] == "ready"
    listed = json.loads(
        text(await execute_library("c2", {"op": "list", "q": "weekly"}, None, None, None))
    )
    assert listed["total"] == 1 and listed["rows"][0]["id"] == asset.id

    store = MetadataStore(isolated_root)
    store.update_fields(asset.id, {"name": "Operator weekly"}, actor="user")

    updated = await execute_library(
        "c3",
        {
            "op": "update",
            "asset_id": asset.id,
            "description": "Weekly numbers, as filed on Mondays.",
            "organized": True,
            "actor": "librarian",
            "confidence": 0.8,
        },
        None,
        None,
        hers,
    )
    assert updated.is_error is False
    row = store.asset(asset.id)
    assert row is not None
    assert row["name"] == "Operator weekly" and row["name_source"] == "user"
    assert row["description_source"] == "librarian"
    assert row["organized_by"] == "librarian" and row["confidence"] == 0.8

    # The user's own protection wins over the agent that runs unattended.
    store.update_fields(asset.id, {"locked": True}, actor="user")
    refused = await execute_library(
        "c4",
        {"op": "update", "asset_id": asset.id, "name": "Renamed", "actor": "librarian"},
        None,
        None,
        hers,
    )
    assert refused.is_error is True
    assert "asset_locked" in text(refused)
    locked = store.asset(asset.id)
    assert locked is not None and locked["name"] == "Operator weekly"

    # The guarded convention writer and the tool's read op agree on the store.
    from local_operator.librarian import profile

    assert profile.record("tags.weekly", "tags", "Weekly reports.", config_dir=isolated_root) == (
        "recorded"
    )
    rows = json.loads(text(await execute_library("c5", {"op": "conventions"}, None, None, None)))
    assert [row["id"] for row in rows["conventions"]] == ["tags.weekly"]


MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class _OneEvalTurn:
    """Answers the first request with an ``eval`` call, then with a plain stop."""

    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests += 1
        first = self.requests == 1

        async def gen():
            if first:
                yield StreamToolCallDelta(
                    index=0, id="c1", name="eval", argument_delta='{"code": "x"}'
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return gen()


@pytest.mark.asyncio
async def test_the_eval_bridge_records_her_write_as_hers(isolated_root, tmp_path) -> None:
    """The route nearly every one of her writes actually took: an eval cell.

    ``tool("library", ...)`` inside a cell is a NESTED dispatch: the loop binds
    the turn's ``ToolContext`` for the cell and copies it with ``dispatch_tool``
    added, so the tool must see the SESSION's id rather than a fresh one. If
    that propagation broke, her own organizing would be recorded as an agent's
    again — which is exactly the bug this file exists to catch — so the cell is
    driven through a real :class:`Session`, a real scripted turn and the real
    registered tool. The cell itself is scripted rather than run on a kernel:
    the seam under test is the context propagation, not the kernel's IPC, and
    ``tool()`` in the eval namespace IS this call.
    """
    from tests.unit.library._fixtures import session as seed_session
    from tests.unit.library._fixtures import write_rows

    sid = await bootstrap.ensure_session(isolated_root)
    assert sid
    report = tmp_path / "weekly.md"
    report.write_text("the Monday numbers", encoding="utf-8")
    seed_session(isolated_root, "feedfacecafe", write_rows("w1", 1.0, report))
    asset = index.build_catalog(isolated_root, budget_s=None).assets[0]
    indexer.reset_for_tests()
    indexer.rebuild_now(isolated_root)

    async def run_cell(tool_call_id, args, signal, on_update, context):
        answer = await context.dispatch_tool(
            "library",
            {"op": "update", "asset_id": asset.id, "name": "Operator weekly"},
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="eval",
            content=[TextContent(text=json.dumps(answer))],
        )

    session = Session(
        model=MODEL,
        stream_fn=_OneEvalTurn(),
        tools=[
            AgentTool(
                name="eval",
                description="eval",
                parameters={"type": "object", "properties": {}},
                approval_tier="read",
                execute=run_cell,
            ),
            build_library_tool(),
        ],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
        session_id=sid,
    )
    # Hermetic: the constructor reads ``tools.defer`` from the isolated config.
    session._tool_deferral = True
    await session.prompt("organize what you have")
    await session.dispose()

    row = MetadataStore(isolated_root).asset(asset.id)
    assert row is not None
    assert row["name"] == "Operator weekly"
    assert row["name_source"] == "librarian", "the bridge lost her identity"
    assert row["organized_by"] == "librarian" and row["organized_at"]
