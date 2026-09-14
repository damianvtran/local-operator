"""The desktop diagnostic reads: wire shape, unknown-vs-zero, hygiene.

These routes exist so the desktop can render the ``/info``, ``/session`` and
``/context`` panels, and every assertion here is made against the HTTP RESPONSE
rather than against a handler's return value. That is not ceremony: the two ways
these routes can ship broken are both invisible in the code that produces them —
a ``dict`` keyed by a TUPLE survives construction and dies at JSON
serialisation, and ``asdict`` silently drops the side attributes that carry
session names and parent edges. A test that called the route function directly
would pass in both cases.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.analytics.store import AnalyticsStore
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.server.routes import capabilities, desktop_catalogues
from tests.unit.analytics.test_store import _snap

TOKEN = "desktop-diagnostics-test-token"
#: Canonical 12-hex session ids — the shape the report route's path parameter is
#: declared against, so a fixture id that could not arrive over HTTP would test
#: something the route never serves.
PARENT = "aabbccddee01"
CHILD = "aabbccddee02"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(capabilities.router)
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()


def _leaves(value: Any):
    """Every scalar literally present in a decoded payload."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def test_context_command_block_carries_the_unformatted_numbers():
    """The `/context` payload a desktop panel reads: rows AND their figures.

    Driven through the handler `POST .../commands` reaches
    (`ServingSessionHandle._context_slash`), not through a helper: the panel
    draws a bar from `data.numbers`, and the rows it also receives are
    pre-formatted strings it MUST NOT parse to recover them.
    """
    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.runtime.serving import ServingSessionHandle

    breakdown = {
        "instructions": 1200,
        "tool_inventory": 300,
        "tool_schemas": 4500,
        "environment": 100,
        "knowledge_mcp_goal": 200,
        "messages": 8000,
        "context_window": 200000,
        "cache_read": 64000,
        "total": 14300,
    }

    class Session:
        def context_breakdown(self):
            return dict(breakdown)

    block = ServingSessionHandle.__new__(ServingSessionHandle)._context_slash(
        Session(), SlashResult
    )
    assert block.kind == "block"
    assert block.data["type"] == "context"
    # Every figure the panel needs, unformatted — the units are TOKENS here and
    # abbreviation is the client's job.
    assert block.data["numbers"] == breakdown
    # The rows are untouched (the terminal reads them), and the total row keeps
    # its human form so the two consumers do not diverge.
    rows = dict(block.data["items"])
    assert rows["Total"] == "~14.3k / 200k (7.1%)"
    # A non-zero cache read adds its own EXACT row, and the number on it still
    # matches `numbers` — one figure, two renderings, never two sources.
    assert rows["Last cache read (exact)"] == "64.0k"


def test_both_hosts_build_the_context_block_from_one_helper():
    """The TUI owner and the detached runtime must answer with one set of numbers.

    A session is owned by one or the other, and `sessions.command` reaches
    whichever happens to hold it, so a second inline copy of the key list in
    `tui/app.py` would let the two hosts disagree about the same command. This
    pins the shared helper rather than either caller's copy of it.
    """
    from local_operator.session.frontend_state import SlashResult, context_block_numbers
    from local_operator.tui.app import OperatorApp

    breakdown = {
        "instructions": 1,
        "tool_inventory": 2,
        "tool_schemas": 3,
        "environment": 4,
        "knowledge_mcp_goal": 5,
        "messages": 6,
        "context_window": 7,
        "cache_read": 8,
        "total": 21,
    }
    assert context_block_numbers(breakdown, 21) == breakdown
    # A breakdown missing a key costs that one figure, never the command.
    assert context_block_numbers({"messages": 6}, 6) == {
        "instructions": 0,
        "tool_inventory": 0,
        "tool_schemas": 0,
        "environment": 0,
        "knowledge_mcp_goal": 0,
        "messages": 6,
        "context_window": 0,
        "cache_read": 0,
        "total": 6,
    }

    app = OperatorApp.__new__(OperatorApp)
    app._context_breakdown = lambda: dict(breakdown)
    block = app._context_slash_result(SlashResult)
    assert block.data["numbers"] == breakdown


async def test_info_route_serves_a_host_snapshot_in_the_reply_envelope(desktop):
    """§5.1: the blocks, the units, and the deliberately EMPTY live half."""
    client, _ = desktop
    response = await client.get("/v1/desktop/info")
    assert response.status_code == 200, response.text
    data = response.json()["result"]["data"]

    assert set(data) == {
        "install",
        "process",
        "sessions",
        "agents",
        "env",
        "degraded",
        "captured_at",
    }
    assert data["process"]["pid"] == os.getpid()
    # epoch SECONDS. The panel subtracts this from ages and uptimes, so a
    # millisecond stamp (~1.7e12) would date the whole host to the year 55,000
    # without failing any type check.
    assert 1e9 < data["captured_at"] < 1e10
    assert abs(time.time() - data["captured_at"]) < 60
    # `degraded` is a list of [name, reason] PAIRS, possibly empty; a reason in
    # this block is how a failed probe is disclosed instead of defaulted away.
    assert isinstance(data["degraded"], list)
    assert all(
        isinstance(row, list) and len(row) == 2 and all(isinstance(part, str) for part in row)
        for row in data["degraded"]
    )
    assert isinstance(data["install"]["version"], str)

    # The live half is NOT MEASURED by this read (`LiveState()` with no session),
    # and the route says so with `None` — the contract's unknown spelling — rather
    # than shipping the dataclass defaults. The panel must never render these:
    # those facts belong to the conversation the desktop already mirrors, and
    # filling them here would give one live fact two sources of truth. Pinned so a
    # later "make the numbers nicer" edit cannot put a confident zero in front of
    # the user; the contract calls that indistinguishability its riskiest
    # assumption.
    assert data["agents"]["tree"] is None
    assert data["agents"]["running"] is None
    assert data["agents"]["queued"] is None
    assert data["agents"]["roster_unread"] is None
    assert data["env"]["mcp_connected"] is None
    assert data["env"]["mcp_settling"] is None
    assert data["env"]["approval_mode"] is None
    assert data["env"]["skills"] is None
    # ... while the HOST half of the very same blocks stays a real reading: a
    # registry scan that finds nothing genuinely measured nothing.
    assert data["agents"]["profiles"] == 0
    assert data["agents"]["teams"] == 0
    assert isinstance(data["sessions"]["total"], int)
    assert isinstance(data["env"]["guides"], int)
    assert isinstance(data["env"]["credential_keys"], list)


async def test_info_route_creates_nothing_on_the_host_it_describes(desktop, tmp_path):
    """A read path must not leave the store it reads behind.

    ``CredentialManager.__init__`` creates the config directory and an empty
    ``credentials.env``, so the credential probe used to WRITE while answering a
    question about a host — the fault class this collector's own comment bans.
    The store is absent before the call and absent after it.
    """
    client, _ = desktop
    credentials = tmp_path / "credentials.env"
    assert not credentials.exists(), "the fixture must not have created the store"

    data = (await client.get("/v1/desktop/info")).json()["result"]["data"]

    assert data["env"]["credential_keys"] == []
    assert not credentials.exists()
    # Scoped to the credential store on purpose: the registry probes in this same
    # snapshot legitimately materialise their OWN directories (``agents/``,
    # ``run/``), which is how those stores work everywhere. The claim here is
    # that the CREDENTIAL probe no longer creates the store it reads.


async def test_analytics_names_are_absent_rather_than_empty(desktop, tmp_path):
    """§5.3: the two side attributes, and what an UNNAMED session looks like.

    No test covered these two fields at all, and their failure mode is silent: a
    rename of the store's ``setattr`` flattens the panel's session tree to hex
    ids with no failing test and no ``degraded`` entry.
    """
    client, _ = desktop
    store = AnalyticsStore(tmp_path / "analytics.db")
    base = replace(_snap(session_id=PARENT), request_id="r1")
    assert (
        store.record_batch(
            [base, replace(base, session_id=CHILD, parent_session_id=PARENT, request_id="r2")]
        )
        == 2
    )
    store.upsert_session_name(PARENT, "Named parent")
    store.close()

    response = await client.get("/v1/desktop/analytics", params={"days": 7})
    assert response.status_code == 200, response.text
    data = json.loads(response.text)["result"]["data"]

    assert data["session_parents"] == {CHILD: PARENT}
    # The UNNAMED session is ABSENT, not present-with-``""``: a client reading
    # ``names?.[id] ?? id`` renders the id only when the key is missing, so an
    # empty string would paint a blank cell styled as an id.
    assert data["session_names"] == {PARENT: "Named parent"}
    assert CHILD not in data["session_names"]
    # The complement pins every map-valued key on this payload, so a breakdown
    # that arrives as an OBJECT fails here even though no list names it.
    assert {key for key, value in data.items() if isinstance(value, dict)} == {
        "aggregate",
        "session_names",
        "session_parents",
    }


async def test_info_route_names_credentials_without_carrying_one(desktop, tmp_path):
    """Secret hygiene: the payload may name a key, never a value or a prefix."""
    client, _ = desktop
    secret = "sk-DIAGPROBE-0123456789abcdef-PREFIX"
    CredentialManager(tmp_path).set_credential("DESKTOP_DIAG_PROBE_KEY", secret, write=True)

    body = (await client.get("/v1/desktop/info")).text
    data = json.loads(body)["result"]["data"]

    # The NAME is the diagnostic answer — "is it even set?" — and the screen is
    # pasted into issues, which is exactly why the value must not be here.
    assert "DESKTOP_DIAG_PROBE_KEY" in data["env"]["credential_keys"]
    assert secret not in body
    assert "DIAGPROBE" not in body
    assert secret[:12] not in body
    # No field OTHER than the key listing may be derived from the store, and
    # every listed entry is a name the manager itself reports.
    known = set(CredentialManager(tmp_path).list_credential_keys())
    assert set(data["env"]["credential_keys"]) <= known
    leaves = [leaf for leaf in _leaves(data) if isinstance(leaf, str)]
    assert not any(secret in leaf for leaf in leaves)
    # The LENGTH is not asserted here on purpose: `len(secret)` is a small
    # integer and this payload is full of small integers (pids, ports, uptimes,
    # rss), so "36 is absent" would be a coincidence test that fails on another
    # machine. The code-level guarantee is narrower and is what was read:
    # `collect_env` reaches the store ONLY through `list_credential_keys()`, and
    # `_multiplexer` consults marker PRESENCE, never a value.


async def test_session_report_route_encodes_tuple_keyed_groups_as_arrays(desktop, tmp_path):
    """§5.2: `by_model`/`by_purpose_outcome` are arrays on the WIRE."""
    client, _ = desktop
    store = AnalyticsStore(tmp_path / "analytics.db")
    base = replace(_snap(session_id=PARENT), request_id="r1", purpose="turn", outcome="stop")
    assert (
        store.record_batch(
            [
                base,
                replace(base, request_id="r2", model_id="other", cost_micro=250),
                replace(base, session_id=CHILD, parent_session_id=PARENT, request_id="r3"),
            ]
        )
        == 3
    )
    store.close()

    response = await client.get(f"/v1/desktop/sessions/{PARENT}/report")
    assert response.status_code == 200, response.text
    # Parsed from the raw text: a tuple key survives the handler and fails only
    # inside json.dumps, so this line is the encoding proof.
    data = json.loads(response.text)["result"]["data"]

    assert isinstance(data["by_model"], list)
    assert {tuple(sorted(row)) for row in data["by_model"]} == {
        ("aggregate", "model_id", "provider")
    }
    assert {(row["provider"], row["model_id"]) for row in data["by_model"]} == {
        ("anthropic", "claude"),
        ("anthropic", "other"),
    }
    assert all(isinstance(row["aggregate"]["calls"], int) for row in data["by_model"])
    assert data["by_purpose_outcome"] == [
        {"purpose": "turn", "outcome": "stop", "calls": 2}
    ]  # OWN scope: the child's call is not folded in, and the edges that let a
    # client re-partition are the only place it appears.
    assert data["aggregate"]["calls"] == 2
    assert data["descendant_ids"] == [CHILD]
    assert data["descendants_aggregate"]["calls"] == 1
    assert set(data["timings"]) == {"duration_ms", "ttft_ms", "preparation_ms"}
    assert len(data["recent"]) == 2


async def test_every_group_by_is_an_array_of_keyed_objects(desktop, tmp_path):
    """One shape for every breakdown, so a FOURTH one added later inherits it.

    Only ``by_model`` and ``by_purpose_outcome`` are forced into arrays — their
    keys are tuples, which JSON cannot carry. ``by_purpose`` is keyed by a
    string and serialised as an OBJECT by accident of its key type, which is not
    a decision about the wire: a response carrying three sibling group-bys in two
    encodings makes the client keep a shape per breakdown.

    The failure this exists to catch is the next group-by being added in the
    object shape and nobody noticing, so it is asserted on the COMPLEMENT as well
    as on a table of the three: a hard-coded table alone cannot see a fourth key
    it has never heard of, while the set of map-valued top-level keys changes the
    moment one arrives as an object.
    """
    client, _ = desktop
    store = AnalyticsStore(tmp_path / "analytics.db")
    base = replace(_snap(session_id=PARENT), request_id="r1", purpose="turn", outcome="stop")
    assert (
        store.record_batch(
            [
                base,
                replace(base, request_id="r2", model_id="other", cost_micro=250),
                replace(base, request_id="r3", purpose="compaction", cost_micro=0),
            ]
        )
        == 3
    )
    store.close()

    response = await client.get(f"/v1/desktop/sessions/{PARENT}/report")
    assert response.status_code == 200, response.text
    data = json.loads(response.text)["result"]["data"]

    shapes = {
        "by_model": ({"provider", "model_id", "aggregate"}, 2),
        "by_purpose": ({"purpose", "aggregate"}, 2),
        "by_purpose_outcome": ({"purpose", "outcome", "calls"}, 2),
    }
    for field, (keys, expected_rows) in shapes.items():
        rows = data[field]
        assert isinstance(rows, list), f"{field} is a {type(rows).__name__}, not an array"
        assert len(rows) == expected_rows, field
        assert all(set(row) == keys for row in rows), field
    assert {row["purpose"] for row in data["by_purpose"]} == {"turn", "compaction"}
    # Order-independent: the store's GROUP BY decides the array order, and this
    # assertion is about the encoding, not about SQLite's sort collation.
    calls_by_purpose = {row["purpose"]: row["aggregate"]["calls"] for row in data["by_purpose"]}
    assert calls_by_purpose == {"turn": 2, "compaction": 1}
    # The complement: the group-bys are the only things that could arrive as
    # objects, so every map-valued key this payload may carry is pinned — a
    # fourth breakdown added later in the object shape fails here even though the
    # table above has never heard of it. ``descendants_aggregate`` is a map when
    # the subtree walk ran and ``null`` when it could not (that is its own
    # contract), and ``timings`` groups three summaries by name.
    assert {key for key, value in data.items() if isinstance(value, dict)} == {
        "aggregate",
        "descendants_aggregate",
        "timings",
    }


async def test_recent_limit_defaults_and_is_clamped_not_rejected(desktop, tmp_path):
    """The 0..50 bound is the STORE's rule; HTTP passes it through unchanged."""
    client, _ = desktop
    store = AnalyticsStore(tmp_path / "analytics.db")
    assert (
        store.record_batch(
            [
                replace(_snap(session_id=PARENT), request_id=f"r{n}", ts_ms=100 + n)
                for n in range(60)
            ]
        )
        == 60
    )
    store.close()

    async def recent(params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        response = await client.get(f"/v1/desktop/sessions/{PARENT}/report", params=params or {})
        assert response.status_code == 200, response.text
        return response.json()["result"]["data"]["recent"]

    assert len(await recent()) == 12
    # Newest first, and the cap holds however far past it a caller asks without
    # turning a documented clamp into an error the panel would have to render.
    rows = await recent({"recent_limit": 3})
    assert [row["request_id"] for row in rows] == ["r59", "r58", "r57"]
    assert len(await recent({"recent_limit": 500})) == 50
    assert await recent({"recent_limit": 0}) == []
    assert await recent({"recent_limit": -1}) == []


async def test_report_distinguishes_unknown_from_zero(desktop, tmp_path):
    """§5.0 on a legacy ledger: absent is not a measured zero."""
    client, _ = desktop
    path = tmp_path / "analytics.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE calls(id INTEGER PRIMARY KEY, ts_ms INTEGER, session_id TEXT, "
            "provider TEXT, model_id TEXT, context_tokens INTEGER, output_tokens INTEGER)"
        )
        conn.execute("INSERT INTO calls VALUES(1, 100, ?, 'p', 'm', 100, 20)", (PARENT,))

    data = (await client.get(f"/v1/desktop/sessions/{PARENT}/report")).json()["result"]["data"]

    assert data["available"] is True
    # No `parent_session_id` column: the subtree walk could not RUN, which is not
    # "$0.00 of subagents".
    assert data["descendants_aggregate"] is None
    assert data["descendant_ids"] == []
    # No tool-call rows (indeed no table): unmeasured, not "0 calls / 0% invalid".
    assert data["tool_calls"] is None
    # Likewise the request row's own unknowns, which must not be drawn as a
    # failure or as zero latency.
    assert data["recent"][0]["ok"] is None
    assert data["recent"][0]["usage_reported"] is None
    assert data["recent"][0]["duration_ms"] is None
    # Nothing here is priceable, so the client renders "—" rather than "$0.00".
    assert data["aggregate"]["cost_known_calls"] == 0
    # An old ledger's absent labels are labelled, not invented.
    assert data["by_purpose_outcome"] == [{"purpose": "unknown", "outcome": "unknown", "calls": 1}]


async def test_report_is_unavailable_not_empty_for_an_unreadable_ledger(desktop, tmp_path):
    """An unopenable store is its own fact; the panel says so rather than "0"."""
    client, _ = desktop
    (tmp_path / "analytics.db").write_text("not sqlite")
    response = await client.get(f"/v1/desktop/sessions/{PARENT}/report")
    assert response.status_code == 200, response.text
    assert response.json()["result"]["data"]["available"] is False


async def test_report_and_info_reject_a_non_canonical_session_id(desktop):
    """The path shape is a route declaration, so it is enforced before the handler."""
    client, _ = desktop
    assert (await client.get("/v1/desktop/sessions/not-a-session/report")).status_code == 422
    # 11 and 13 hex characters are both rejected: the length is part of the id's
    # shape, not a formatting nicety.
    assert (await client.get("/v1/desktop/sessions/aabbccddee0/report")).status_code == 422
    assert (await client.get("/v1/desktop/sessions/AABBCCDDEE01/report")).status_code == 422


async def test_diagnostics_capability_is_advertised_and_both_ops_are_gated(desktop):
    """§5.6: one new key, and the plane's own boundary on both routes."""
    client, _ = desktop
    features = (await client.get("/v1/capabilities")).json()["result"]["features"]
    assert features["diagnostics"] == 1
    # A missing key means "this backend has no such route": an older backend
    # cannot be identified by version, so the key is the whole negotiation.
    assert features["catalogues"] == 1

    for path in ("/v1/desktop/info", f"/v1/desktop/sessions/{PARENT}/report"):
        assert (await client.get(path)).status_code == 200, path
        assert (await client.get(path, headers={"Authorization": ""})).status_code == 401, path
        assert (
            await client.get(path, headers={"Authorization": "Bearer wrong"})
        ).status_code == 401, path


def test_info_registry_row_carries_the_panel_destination():
    """`/info` is offered on the desktop now, and its row says where it goes."""
    from local_operator.server.utils.desktop_commands import (
        OWNER_COMMANDS,
        command_catalogue,
    )
    from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for

    spec = slash_command_for("/info")
    assert spec is not None
    assert spec.desktop_destination == "info"
    # Read-only host view: no owner execution, so no execution handler to add.
    assert spec.name not in OWNER_COMMANDS
    catalogued = command_catalogue()
    row = next(item for item in catalogued if item["name"] == "info")
    assert row["destination"] == "info"
    assert row["execution"] == "native"
    # The aliases stay absent (their removal is its own recorded decision).
    assert row["aliases"] == []
    # Parity with the registry: the catalogue is exactly the rows that carry a
    # destination, in registry order, which is what the transport's own record
    # claims and what `/mobile` alone is now excluded from.
    assert [item["name"] for item in catalogued] == [
        item.name for item in SLASH_COMMANDS if item.desktop_destination
    ]
