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
from local_operator.librarian import bootstrap, proactive


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
    report = tmp_path / "weekly.md"
    report.write_text("the Monday numbers", encoding="utf-8")
    seed_session(isolated_root, "feedfacecafe", write_rows("w1", 1.0, report))
    asset = index.build_catalog(isolated_root, budget_s=None).assets[0]

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
        None,
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
        None,
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
