"""The server side of a CONFIGURATION RUN: an unlisted, watchable, bounded session.

The feature is one additive field (``CreateSession.purpose``) and what the
backend does with it. Every property below is one a later edit can break
silently, which is why each has a test that can fail rather than a sentence:

* **hidden, but not unreachable.** The run's origin (``agent-config``) is kept
  out of ``USER_ORIGINS``, so it is absent from the catalogue, ``/resume``, the
  phone list, the attention feed and the first-run scan — and it is admitted
  through the desktop DOOR, because the page watches the run by id. The door
  widening must admit exactly this origin: a ``subagent`` session is still
  refused.
* **origin before marker.** The origin stamp lands before ``desktop.json``,
  because the marker is what materialises the session and the listing surfaces
  memoise the user-session verdict per id.
* **the server owns the run's shape.** The client sends no cwd, no model, no
  target and no draft; a peer + purpose is refused; a second create while a run
  is live answers 409 carrying the active run's id.
* **the bounded inventory is by OP, not only by name.** ``agent reset`` and
  ``agent sync`` are ``op=`` values of the single ``agent`` tool, so declaring
  the tool is not enough.

NO TEST HERE SPAWNS A RUNTIME: a create writes a marker and starts nothing, and
the runtime record the single-flight read consumes is written by hand.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import ModelSpec
from local_operator.prompts_api import build_system_blocks, split_tail_sections
from local_operator.resume import (
    ORIGIN_AGENT_CONFIG,
    ORIGIN_SUBAGENT,
    is_user_session,
    mark_session_origin,
    session_origin,
)
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils import desktop_sessions as pool_module
from local_operator.server.utils.desktop_sessions import (
    DesktopSessions,
    read_desktop_marker,
)
from local_operator.session.retention import AGENTS_CONFIG_PURPOSE, read_desktop_purpose
from local_operator.session.runtime.types import SessionRecord

MODEL = ModelSpec(provider="test", model_id="m", context_window=1000)


def _create_body(root: Path, **extra: Any) -> dict[str, Any]:
    """A valid create body; ``request_id`` fresh unless overridden."""
    return {"request_id": str(uuid.uuid4()), "cwd": str(root), **extra}


def _never_streams(request: Any, signal: Any) -> Any:
    """A ``stream_fn`` that fails loudly: none of these tests runs a turn.

    A plain ``def`` returning the generator, not an ``async def``: the parameter
    is an async-GENERATOR function, and a coroutine function is a different type
    (a coroutine is not an ``AsyncIterator``, which is what pyright reports).
    """

    async def gen():
        raise AssertionError("no turn is run here")
        yield

    return gen()


@pytest_asyncio.fixture
async def config_app(tmp_path: Path, monkeypatch):
    """A minimal app over THIS test's config root, with the pool ATTACHED.

    The shape ``test_desktop_drafts.py`` established (own ``tmp_path``, every
    ``CMUX_*`` stripped, the pool closed at the end), kept identical so a change
    to the wiring shows up in both suites rather than in one.
    """
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "[redacted]")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer [redacted]"},
    ) as client:
        yield client, app, tmp_path.resolve()
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def _write_live_record(root: Path, session_id: str, *, busy: bool) -> None:
    """Publish a live runtime record for ``session_id`` by hand.

    ``os.getpid()`` so ``registry.scan`` classifies it ``live`` (a numeric pid that
    does not exist is reaped as stale, which would make this test vacuous), and
    ``busy`` because that is the term the single-flight read turns on.
    """
    record = SessionRecord.from_json(
        {
            "pid": os.getpid(),
            "kind": "daemon",
            "session_id": session_id,
            "conversation_name": "synthetic",
            "cwd": str(root),
            "model_label": "synthetic-model",
            "control_port": 1,
            "control_key": "k",
            "busy": busy,
        }
    )
    directory = root / "run" / "mobile"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{record.pid}.json").write_text(json.dumps(record.to_json()), encoding="utf-8")


# ---------------------------------------------------------------------------
# The create: the hidden origin, and the ordering that makes it stick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_purpose_create_hides_the_run_and_records_it(config_app) -> None:
    """The run is stamped, unlisted, and carries its purpose in the marker.

    Three facts at once, because they are one decision: the origin keeps it out
    of every listing (``is_user_session`` is what five surfaces read), the marker
    key is what the RUNTIME reads to learn it is a run, and the cwd is the config
    root the caller never named.
    """
    client, app, root = config_app
    response = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "purpose": "agents-config"},
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["result"]["session_id"]
    directory = root / "sessions" / session_id

    assert session_origin(directory) == ORIGIN_AGENT_CONFIG
    assert is_user_session(directory) is False, "a run must not be listed as the operator's own"
    assert read_desktop_purpose(directory) == AGENTS_CONFIG_PURPOSE
    marker = read_desktop_marker(directory)
    assert marker is not None and marker["cwd"] == str(root), "the server names the folder"
    assert (directory / "desktop.json").is_file()


@pytest.mark.asyncio
async def test_the_origin_stamp_lands_before_the_marker(tmp_path: Path, monkeypatch) -> None:
    """The ORDER is the property, so the assertion is taken AT the marker write.

    A spy rather than a post-hoc file read: by the time ``create`` returns, both
    files exist whichever order they were written in, so only an observation
    inside the write can tell the two apart. What it records is the fact the
    design turns on — at the instant the marker is published, the origin is
    already readable (a scan or a feed tick in the gap would otherwise cache a
    WRONG verdict, and the cache is what makes the mistake permanent).
    """
    root = tmp_path
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(root))
    observed: list[str] = []
    real = pool_module.write_desktop_marker

    def spy(path, directory, *, model=None, purpose=None):
        observed.append(session_origin(path))
        observed.append("marker-first" if not (path / "desktop.json").exists() else "marker-there")
        return real(path, directory, model=model, purpose=purpose)

    monkeypatch.setattr(pool_module, "write_desktop_marker", spy)
    pool = DesktopSessions(root)
    session_id = await pool.create(str(root), purpose=AGENTS_CONFIG_PURPOSE)
    assert observed == [ORIGIN_AGENT_CONFIG, "marker-first"], (
        "desktop.json materialises the session, so the origin stamp must already be on "
        f"disk when it is written (observed {observed})"
    )
    assert session_origin(root / "sessions" / session_id) == ORIGIN_AGENT_CONFIG


# ---------------------------------------------------------------------------
# The door: one origin wider, and no wider
# ---------------------------------------------------------------------------


def test_the_door_admits_a_run_and_still_refuses_a_subagent(tmp_path: Path) -> None:
    """The widening is exactly one origin — asserted in both directions.

    The negative half is the load-bearing one: a rewrite to "hidden but
    reachable" would admit every machine-made session in the store, and this is
    the test that fails when someone makes that edit.
    """
    run = tmp_path / "sessions" / "aaaa00000001"
    run.mkdir(parents=True)
    mark_session_origin(run, ORIGIN_AGENT_CONFIG)
    child = tmp_path / "sessions" / "aaaa00000002"
    child.mkdir(parents=True)
    mark_session_origin(child, ORIGIN_SUBAGENT)
    user = tmp_path / "sessions" / "aaaa00000003"
    user.mkdir(parents=True)

    assert pool_module._door_may_open(run) is True
    assert pool_module._door_may_open(user) is True
    assert (
        pool_module._door_may_open(child) is False
    ), "the widening must not generalise to every hidden origin"


@pytest.mark.asyncio
async def test_a_run_is_reachable_by_the_desktop_doors(config_app) -> None:
    """Hidden from listings, reachable by id — the two halves of one decision.

    The snapshot route is the whole door in one call: if ``locate`` refused, the
    page could not watch, prompt or interrupt the run it started.
    """
    client, app, root = config_app
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "purpose": "agents-config"},
    )
    session_id = created.json()["result"]["session_id"]
    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    # AND THE OTHER HALF, through the surface a user actually reads: the
    # catalogue the sidebar polls must not carry the run. Asserted here rather
    # than only through ``is_user_session`` because the predicate is shared and
    # this is the reader that would show the leak.
    listing = await client.get("/v1/desktop/sessions?limit=50")
    assert listing.status_code == 200, listing.text
    rows = listing.json()["result"]["sessions"]
    assert session_id not in {
        row["session_id"] for row in rows
    }, "a configuration run must not appear in the operator's conversation list"


# ---------------------------------------------------------------------------
# Single flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_flight_is_a_live_busy_run(config_app) -> None:
    """The read that answers both "may this create proceed" and "re-attach here".

    ``busy`` rather than "resident", and the two cases are both asserted: a run
    whose turn is in flight is the run a second create must join; one whose
    runtime is alive but idle is not a race, so the page's "start a new one" is a
    real action rather than one the previous run must be stopped to allow.
    """
    client, app, root = config_app
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "purpose": "agents-config"},
    )
    session_id = created.json()["result"]["session_id"]
    pool: DesktopSessions = app.state.desktop_sessions

    assert await _offload(pool.active_config_run) is None, "no record yet means no live run"

    _write_live_record(root, session_id, busy=False)
    assert (
        await _offload(pool.active_config_run) is None
    ), "a resident but idle run is not what a second run would race"

    _write_live_record(root, session_id, busy=True)
    assert await _offload(pool.active_config_run) == session_id


@pytest.mark.asyncio
async def test_a_second_create_answers_409_with_the_active_run(config_app) -> None:
    """A second window JOINS the run instead of starting a rival.

    The id has to be in the refusal: without it the client can only say "already
    running", which is the sentence that makes a user close the wrong window.
    """
    client, app, root = config_app
    first = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "purpose": "agents-config"},
    )
    session_id = first.json()["result"]["session_id"]
    _write_live_record(root, session_id, busy=True)

    second = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "purpose": "agents-config"},
    )
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["code"] == "agents_config_running"
    assert detail["session_id"] == session_id


async def _offload(fn: Any) -> Any:
    """Run a synchronous pool read the way the route does (a worker hop)."""
    import asyncio

    return await asyncio.to_thread(fn)


# ---------------------------------------------------------------------------
# The shape the client may not supply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peer_create_with_a_purpose_is_refused(config_app) -> None:
    """Local-only, with its OWN code: the two refusals say different things."""
    client, app, root = config_app
    response = await client.post(
        "/v1/desktop/sessions",
        json={
            "request_id": str(uuid.uuid4()),
            "purpose": "agents-config",
            "peer": "aaaaaaaaaaaa",
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "agents_config_local_only"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, value",
    [
        ("cwd", "/tmp"),
        ("model", {"provider": "test", "model_id": "m"}),
        ("target", {"kind": "agent", "name": "reviewer"}),
        ("draft_id", "abcdefabcdef"),
    ],
)
async def test_client_supplied_shape_fields_are_refused(config_app, field, value) -> None:
    """Refused rather than ignored, and refused with the field's name in it.

    Tolerance would let a renderer believe it had chosen a model for a run that
    used none — and a model is a PAID turn, so the belief costs money.
    """
    client, app, root = config_app
    response = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "purpose": "agents-config", field: value},
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "agents_config_client_fields"
    assert field in detail["message"]


@pytest.mark.asyncio
async def test_the_capability_key_is_published(config_app) -> None:
    """The renderer gates the whole composer on this key, so it must be present."""
    client, app, root = config_app
    response = await client.get("/v1/capabilities")
    features = response.json()["result"]["features"]
    assert features["agents_config"] == 1
    # The run needs these to be stoppable and to show its results; the client
    # requires the conjunction rather than a second spelling of the same fact.
    for needed in ("session_interrupt", "profile_catalogue", "team_catalogue"):
        assert features[needed] >= 1


@pytest.mark.asyncio
async def test_an_ordinary_create_still_needs_a_folder(config_app) -> None:
    """The carve-out is the RUN's, not the route's: nothing loosens for anyone else."""
    client, app, root = config_app
    response = await client.post(
        "/v1/desktop/sessions", json={"request_id": str(uuid.uuid4()), "cwd": ""}
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# The run's own shape: the inventory, its OP scope, and the preamble
# ---------------------------------------------------------------------------


def _op_tool(name: str, ops: list[str], calls: list[str]) -> Any:
    """A stand-in for a multiplexing builtin: one ``op`` enum, one recorder."""

    async def execute(tool_call_id, args, signal=None, on_update=None, context=None):
        from local_operator.harness.types import TextContent, ToolResult

        calls.append(str(args.get("op")))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=name,
            content=[TextContent(text="ran")],
        )

    from local_operator.harness.types import AgentTool

    return AgentTool(
        name=name,
        parameters={
            "type": "object",
            "properties": {"op": {"type": "string", "enum": ops}},
        },
        execute=execute,
    )


def _session_with(tmp_path: Path, tools: list[Any]) -> Any:
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    return Session(
        model=MODEL,
        stream_fn=_never_streams,
        tools=tools,
        transcript=Transcript(tmp_path / f"session-{uuid.uuid4().hex[:8]}"),
        system_blocks_provider=lambda: ["sys"],
    )


@pytest.mark.asyncio
async def test_the_declaration_bounds_the_run_by_name_and_by_op(tmp_path: Path) -> None:
    """`reset`/`sync` are ``op=`` values of the ``agent`` tool, so names are not enough.

    Both halves are asserted because either alone is a half-answer: the SCHEMA the
    model is shown must not advertise an op the run cannot use, and a call that
    names one anyway must be refused before the tool's own dispatch — that second
    half is the property the declaration sells ("not reachable", not "not told").
    """
    from local_operator.session_factory import (
        AGENTS_CONFIG_TOOL_OPS,
        AGENTS_CONFIG_TOOLS,
    )

    agent_calls: list[str] = []
    team_calls: list[str] = []
    session = _session_with(
        tmp_path,
        [
            _op_tool("agent", ["list", "reset", "sync", "create"], agent_calls),
            _op_tool("team", ["list", "create"], team_calls),
            _op_tool("team_delete", ["delete"], []),
        ],
    )
    session.set_tool_inventory(AGENTS_CONFIG_TOOLS, ops=AGENTS_CONFIG_TOOL_OPS)

    assert {tool.name for tool in session._tools} == {"agent", "team"}, "team_delete must go"
    schema = next(t for t in session._tools if t.name == "agent").parameters
    assert "reset" not in schema["properties"]["op"]["enum"]
    assert "sync" not in schema["properties"]["op"]["enum"]
    assert "create" in schema["properties"]["op"]["enum"]

    agent = next(t for t in session._tools if t.name == "agent")
    refused = await agent.execute("call-1", {"op": "sync"})
    assert refused.is_error, "a sync call must be refused by the declaration"
    assert agent_calls == [], "the refusal must happen BEFORE the tool's own dispatch"
    allowed = await agent.execute("call-2", {"op": "list"})
    assert not allowed.is_error
    assert agent_calls == ["list"]


def test_the_op_scope_is_one_way(tmp_path: Path) -> None:
    """A later call may narrow a tool's ops but never widen them.

    The hole this closes is otherwise invisible: the second call is the one place
    a run's op scope could be lifted after the fact, and the names' invariant
    would read as though it covered this too.
    """
    from local_operator.session_factory import (
        AGENTS_CONFIG_TOOL_OPS,
        AGENTS_CONFIG_TOOLS,
    )

    session = _session_with(tmp_path, [_op_tool("agent", ["list", "sync"], [])])
    session.set_tool_inventory(AGENTS_CONFIG_TOOLS, ops=AGENTS_CONFIG_TOOL_OPS)
    with pytest.raises(ValueError, match="op scope is one-way"):
        session.set_tool_inventory(AGENTS_CONFIG_TOOLS, ops={"agent": ("list", "sync")})


def test_applying_the_run_shape_needs_the_marker(tmp_path: Path) -> None:
    """The shape follows the MARKER, so it holds for a run resumed hours later.

    Returns a bool so the test asserts on a public path's answer rather than on
    private state, and is a no-op for every session without the marker — which is
    every ordinary conversation.
    """
    from local_operator.session.retention import DESKTOP_PURPOSE_KEY
    from local_operator.session_factory import (
        AGENTS_CONFIG_PREAMBLE,
        apply_config_run_shape,
    )

    plain = _session_with(tmp_path, [_op_tool("agent", ["list"], [])])
    assert apply_config_run_shape(plain) is False
    assert plain._goal_state.run_brief == ""

    run_dir = tmp_path / "sessions" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "desktop.json").write_text(
        json.dumps(
            {"version": 1, "cwd": str(tmp_path), DESKTOP_PURPOSE_KEY: AGENTS_CONFIG_PURPOSE}
        ),
        encoding="utf-8",
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    run = Session(
        model=MODEL,
        stream_fn=_never_streams,
        tools=[_op_tool("agent", ["list", "sync"], []), _op_tool("bash", ["run"], [])],
        transcript=Transcript(run_dir),
        system_blocks_provider=lambda: ["sys"],
    )
    assert apply_config_run_shape(run) is True
    assert run._goal_state.run_brief == AGENTS_CONFIG_PREAMBLE
    assert {tool.name for tool in run._tools} == {"agent"}, "the declaration must be in force"


def test_the_preamble_rides_the_tail_and_round_trips() -> None:
    """It is its OWN section, so a delta can re-ship it alone and a resume replays it.

    The round-trip is the contract ``split_tail_sections`` states: a marker whose
    split does not reassemble would rewrite the state the next delta compares
    against.
    """
    blocks = build_system_blocks(
        [],
        "<skills/>",
        "env",
        "2026-09-30",
        run_brief="You author agents and teams.",
    )
    tail = blocks[3]
    assert "<configuration-run>" in tail
    sections = split_tail_sections(tail)
    assert sections.get("run") == (
        "<configuration-run>\nYou author agents and teams.\n</configuration-run>"
    )
    from local_operator.prompts_api import assemble_system_block_sections

    assert assemble_system_block_sections(3, sections) == tail
