"""The viewer→runtime attach, over a real socket, end to end.

**This file exists because the whole suite stayed green through a total
product outage.** Round 1 QA and UX both found, against the real binary, that
no message could be sent in any session on this branch: `RuntimeServer`
advertises `tui_state_v1` only when its handle has `subscribe_frontend`, that
method lived only on the owner-path class this PR deletes, so every runtime
published `capabilities: []` and hung up on every viewer.

189 unit tests and 4 e2e tests passed anyway, for two structural reasons:

* `tests/unit/session/runtime/test_server.py`'s stub handle DECLARES
  `subscribe_frontend`, so the server tests exercise a capability the
  production handle did not have; and
* `tests/e2e/test_tui_e2e.py` injects an in-process `Session` straight into
  `OperatorApp`, so it never performs an attach at all.

Both are reasonable in isolation and together they left the seam this release
is *about* with no coverage. So the rule these tests follow is: **the
production handle class, the production server, a real loopback socket, and
the production `RemoteSession` client.** Nothing here may substitute a stub
for the object under test — a stub that declares the capability is precisely
what hid the defect.

Kept out of `tests/unit` deliberately: it binds a socket and drives two
asyncio components against each other, which is the e2e stage's job.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.session.runtime import registry
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


async def _never_take_over() -> Any:
    raise AssertionError("a viewer must never take over a session")


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 10.0) -> Any:
    """The record the runtime publishes, once it is discoverable."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record published for {session_id} within {timeout}s")


async def _runtime(
    directory: Path, replies: list[str]
) -> tuple[Any, OwnedSessionHandle, RuntimeServer]:
    """A real Session behind the production handle behind the production server."""

    # ScriptedStream is the harness's own provider double: one async
    # generator per model call, the same shape the session's real provider
    # contract has. A bare list is not iterable with `async for`.
    stream = ScriptedStream([text_turn(reply) for reply in replies] or [text_turn("ok")])
    session = build_session(directory, stream)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    return session, handle, server


@pytest.mark.asyncio
async def test_the_runtime_advertises_the_full_tui_capability(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The one-line regression guard for the round-1 blocker.

    `capabilities: []` is what refused every viewer. Asserted on the RECORD
    the server publishes rather than on the handle, because the record is what
    a client reads and `server.py` gates the handshake on it.
    """
    directory = headless_tui_env / "sessions" / "capsess00001"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, [])
    await server.start_in_process()
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        assert "tui_state_v1" in record.capabilities, (
            "the runtime must advertise the full-TUI capability; without it "
            "server.py hangs up on every viewer and no message can be sent"
        )
    finally:
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_viewer_attaches_over_a_real_socket_and_runs_a_turn(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The full path the product depends on: attach, prompt, stream, persist.

    This is the test whose absence let the outage ship. It performs the exact
    handshake `RemoteSession` performs in production (`frontend_state=True`),
    against the exact handle the runtime builds, and then drives a turn to a
    durable transcript row.
    """
    from local_operator.session.remote import RemoteSession

    directory = headless_tui_env / "sessions" / "attachsess01"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["Hello from the runtime."])
    await server.start_in_process()
    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)

        # The production client, asking for what the production viewer asks
        # for. Before the fix this raised ConnectionError("owner closed the
        # connection") right here.
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )

        assert viewer.frontend_state is not None, "the attach carried no state seed"

        await viewer.prompt("hello there")

        # The turn ran on the RUNTIME and reached its durable transcript,
        # which is the property a viewer cannot fake: the reply is not in the
        # viewer's memory, it is on disk in the runtime's session directory.
        transcript_path = directory / "transcript.jsonl"
        for _ in range(200):
            if transcript_path.exists() and "Hello from the runtime." in transcript_path.read_text(
                encoding="utf-8"
            ):
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - only on a real regression
            written = (
                transcript_path.read_text(encoding="utf-8")
                if transcript_path.exists()
                else "<absent>"
            )
            raise AssertionError(f"the runtime never wrote the assistant reply; got {written}")
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_event_relay_reaches_an_attached_viewer(
    headless_tui_env: Path, workspace: Path
) -> None:
    """`subscribe_events` is the other half of the capability.

    Its absence is quieter than the handshake failure and would survive a fix
    that only restored `subscribe_frontend`: the attach succeeds, and then
    nothing ever streams. Asserted on events the VIEWER received, so it fails
    if the relay is wired but not delivering.
    """
    from local_operator.session.remote import RemoteSession

    directory = headless_tui_env / "sessions" / "relaysess001"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["Streamed reply."])
    await server.start_in_process()
    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )
        seen: list[Any] = []
        viewer.subscribe(seen.append)

        await viewer.prompt("say something")

        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.05)
        assert seen, "no AgentEvent reached the viewer; the v4 relay is dead"
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_viewer_runs_a_team_and_holds_a_credential(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The two capabilities a viewer silently lacked, over the real socket.

    Both were reported by the operator as "nothing happens". They share one
    mechanism — state that lives on ``Session`` and has no seam on
    ``RemoteSession`` — and both are asserted here on the OWNER's state, which
    is the half a viewer cannot fake.

    ``/team``: the mutating form used to return an unconsumed
    ``noop {"type": "team_mutate"}``, so the command evaporated. The property
    is that the attach lands on the session that builds the next turn.

    ``/credential``: the store is an in-memory per-process dict whose reader is
    ``credential_env()`` inside the `bash` tool — which runs HERE. A viewer
    holding its own store would satisfy a naive round-trip test and still leave
    every tool unable to read the secret, so this asserts the value arrives in
    a real child process's environment. Never by printing it: the length is
    proof enough and the value must not enter a log or a transcript.
    """
    from local_operator.session.remote import RemoteSession
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    registry = TeamRegistry(headless_tui_env)
    registry.create_team(
        TeamEditFields(
            name="viewerteam",
            description="d",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )

    directory = headless_tui_env / "sessions" / "capgapsess1"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["ack"])
    session.team_registry = registry
    await server.start_in_process()
    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )

        outcome = await viewer.route_shared_slash("team", "viewerteam do the thing", [])
        assert outcome.get("kind") != "noop", (
            "the mutating /team returned an unconsumed noop again; the command "
            "renders nothing at all on a viewer"
        )
        assert (outcome.get("data") or {}).get("type") == "team_attached"
        assert (outcome.get("data") or {}).get("request") == "do the thing"
        assert session.active_team_name == "viewerteam", (
            "the attach must land on the OWNER, which is where the roster and "
            "briefs are stamped onto the turn"
        )

        secret = "abcd-1234-efgh"
        stored = await viewer.credential_op("store", "E2E_TOKEN", secret)
        assert stored.get("ok"), stored
        assert secret not in str(stored), "the value must never travel back"
        assert session.variables.credential_names() == ["E2E_TOKEN"], (
            "the credential must land in the OWNER's store — that is the one "
            "the bash tool reads through credential_env()"
        )

        from local_operator.tools.builtin import execute_bash

        class _Ctx:
            variables = session.variables
            cwd = str(directory)

        result = await execute_bash(
            "e2e-cred-probe",
            {"command": 'test -n "$E2E_TOKEN" && echo LEN=${#E2E_TOKEN}'},
            None,
            None,
            cast("Any", _Ctx()),
        )
        rendered = str(getattr(result, "content", result))
        assert (
            f"LEN={len(secret)}" in rendered
        ), f"the credential never reached the tool's environment: {rendered}"
        assert secret not in rendered, "the value must not appear in tool output"
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", ["", "ship it"], ids=["bare-attach", "with-request"])
async def test_a_cold_routed_team_command_is_not_retired_by_an_immediate_quit(
    headless_tui_env: Path, workspace: Path, request_text: str
) -> None:
    """#622 × #624: the runtime a cold `/team x` just engaged must survive the quit.

    Since #622 a viewer that leaves offers its runtime back
    (``retire_if_pristine``) and the runtime refuses when anything durable
    exists. #624 makes a cold viewer engage a runtime and route ``/team
    <name> [<request>]`` through it. The sequence to rule out: cold `/team x`
    → `ctrl+d` → the runtime reads as pristine → it retires, discarding the
    attachment the "team x is ready" receipt just vouched for and stranding a
    sidecar-only directory.

    The BARE form is the primary cell (review round 2, R7). The first version
    of this test used the request form only and passed for the wrong reason:
    the viewer sends the ``prompt`` frame before the quit, and that — not the
    attach — was what made the session non-pristine. With no request there is
    no prompt frame, so the attach's own durability (the ``attachment.json``
    sidecar, which ``is_pristine`` now consults) is the only thing standing
    between the runtime and retirement.

    Driven end to end with nothing stubbed on the runtime side: a production
    ``OwnedSessionHandle`` behind a production ``RuntimeServer`` (the same
    ``is_pristine`` probe and the same refusal path a real child runs), a real
    ``RemoteSession.cold(<minted id>)`` whose ``engage_runtime`` is pointed at
    that server, the real ``OperatorApp`` submitting the command as one paste
    + Enter and then quitting on the very next tick.
    """
    import os
    import uuid

    from textual import events

    from local_operator.session.remote import RemoteSession
    from local_operator.session.runtime import launch as launch_module
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor

    (headless_tui_env / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    TeamRegistry(headless_tui_env).create_team(
        TeamEditFields(
            name="quitteam", description="d", manager="manager", members=[TeamMember(role="coder")]
        )
    )
    session_id = uuid.uuid4().hex[:12]
    directory = headless_tui_env / "sessions" / session_id
    directory.mkdir(parents=True)
    # A slow reply keeps the turn IN FLIGHT across the quit — the window the
    # retirement decision must respect.
    session, handle, server = await _runtime(directory, ["ack", "ack"])
    session.team_registry = TeamRegistry(headless_tui_env)
    retire_details: list[str] = []
    real_retire = server._retire_if_pristine

    async def spy_retire(*, leaving: Any) -> str:
        detail = await real_retire(leaving=leaving)
        retire_details.append(detail)
        return detail

    server._retire_if_pristine = spy_retire  # type: ignore[method-assign]

    async def engage_here(sid: str, cwd: str, work: Any, *, config_dir: Path, deadline_s=30.0):
        # Stand in for the spawn only: the record, the dial and the sync are
        # production code against this real server.
        await server.start_in_process()
        (directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
        await _wait_for_record(config_dir, sid)
        return None

    original = launch_module.engage_runtime
    launch_module.engage_runtime = engage_here  # type: ignore[assignment]
    try:

        async def factory() -> RemoteSession:
            return await RemoteSession.cold(
                session_id,
                config_dir=headless_tui_env,
                cwd=str(directory),
                takeover_factory=_never_take_over,
            )

        OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(200):
                await pilot.pause()
                if app._session is not None:
                    break
            viewer = app._session
            assert isinstance(viewer, RemoteSession)
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            line = f"/team quitteam {request_text}".rstrip()
            app.post_message(events.Paste(line))
            await pilot.pause()
            await pilot.press("enter")
            if not request_text:
                # A bare name parks as `/team quitteam ` with the name-argument
                # picker's completion; the second Enter is the blank-Enter
                # SWITCH the picker advertises ("Enter to switch").
                await pilot.pause()
                await pilot.press("enter")
            # Wait only for the ROUTE to land (the attach is synchronous on the
            # owner inside the slash handler), then quit at once.
            for _ in range(400):
                await pilot.pause()
                if session.active_team_name == "quitteam":
                    break
                await asyncio.sleep(0.01)
            assert session.active_team_name == "quitteam", "the cold route never attached"
            app.exit()
        # Quit ran `_retire_unused_runtime` before dispose.
        assert retire_details, "the viewer never offered the runtime back"
        assert retire_details[-1].startswith("kept:"), retire_details
        assert not handle.is_pristine(), "a stamped team is durable state, not a pristine session"
        from local_operator.resume import (
            ATTACHMENT_SIDECAR_NAME,
            read_session_attachment,
        )

        assert (directory / ATTACHMENT_SIDECAR_NAME).exists(), "the attachment must survive"
        restored = read_session_attachment(directory)
        assert restored is not None and getattr(restored, "team", "") == "quitteam", restored
        assert session.active_team_name == "quitteam", "the runtime kept the team it was given"
        if not request_text:
            # The sidecar ALONE held the runtime: no prompt frame, no row.
            assert not (directory / "transcript.jsonl").exists(), "the bare form wrote a row"
        if request_text:
            # The runtime was NOT stopped: the turn it was given still lands.
            deadline = asyncio.get_running_loop().time() + 15
            body = ""
            while asyncio.get_running_loop().time() < deadline:
                body = (
                    (directory / "transcript.jsonl").read_text()
                    if (directory / "transcript.jsonl").exists()
                    else ""
                )
                if request_text in body and "ack" in body:
                    break
                await asyncio.sleep(0.05)
            assert request_text in body and "ack" in body, f"the turn was lost:\n{body}"
    finally:
        launch_module.engage_runtime = original  # type: ignore[assignment]
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_an_undeclaring_client_has_its_team_request_run_by_the_runtime(
    headless_tui_env: Path, workspace: Path
) -> None:
    """THE INCIDENT, reproduced on the wire and then repaired.

    A raw ``AttachClient`` constructed WITHOUT ``slash_consumers`` is exactly
    what a pre-#624 viewer looks like to a runtime: the auth frame carries no
    declaration, because that build had no such field. This was verified live
    against a shipped 0.49.0 runtime — the probe received the ``team_attached``
    receipt with the request inside it, and the session's transcript recorded
    no user row and no turn. The team was attached; the request evaporated.

    So this cell asserts on the OWNER's session, which is the half a client
    cannot fake: the attach landed AND the request ran as a real turn. Driven
    through the production handle, the production server and a real loopback
    socket, because a stub that declares the capability is exactly what hid
    the original defect (see this module's docstring).
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    registry = TeamRegistry(headless_tui_env)
    registry.create_team(
        TeamEditFields(
            name="viewerteam",
            description="d",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )

    directory = headless_tui_env / "sessions" / "skewsess0001"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["ack"])
    session.team_registry = registry
    await server.start_in_process()
    client = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)

        # No ``slash_consumers``: the old viewer on the wire.
        client = AttachClient(lambda _p: None, lambda _r: None)
        await client.connect(record, session.session_id)

        outcome = await client.slash_result("team", "viewerteam do the thing", [])

        assert (outcome.get("data") or {}).get("type") == "team_attached"
        assert session.active_team_name == "viewerteam", "the attach must still land"

        # THE PROPERTY THE INCIDENT LACKED: a real turn ran. Asserted on the
        # runtime's durable transcript rather than on any client-side echo,
        # because the transcript is what the manager's live probe read to
        # prove the request had been dropped.
        transcript_path = directory / "transcript.jsonl"
        for _ in range(200):
            if transcript_path.exists() and "do the thing" in transcript_path.read_text(
                encoding="utf-8"
            ):
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - only on a real regression
            written = (
                transcript_path.read_text(encoding="utf-8")
                if transcript_path.exists()
                else "<absent>"
            )
            raise AssertionError(
                "the runtime never ran the request an undeclaring client cannot "
                f"submit itself; transcript was {written}"
            )
    finally:
        if client is not None:
            client.close()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_declaring_client_is_never_double_submitted(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The guard on the repair: a current viewer submits its own request.

    ``SLASH_ACTION_RECEIPTS`` in the auth frame is a promise — "I render these
    and will submit the request myself". A runtime that admitted anyway would
    turn one typed command into two turns, which is a worse failure than the
    silent drop because the user cannot tell which turn is which or undo one.

    Asserted as the ABSENCE of a turn on the owner, with the attach present:
    the receipt still comes back, the roster is still stamped, and the
    transcript stays empty of the request.
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.types import SLASH_ACTION_RECEIPTS
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    registry = TeamRegistry(headless_tui_env)
    registry.create_team(
        TeamEditFields(
            name="viewerteam",
            description="d",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )

    directory = headless_tui_env / "sessions" / "skewsess0002"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["ack"])
    session.team_registry = registry
    await server.start_in_process()
    client = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        client = AttachClient(
            lambda _p: None,
            lambda _r: None,
            slash_consumers=list(SLASH_ACTION_RECEIPTS),
        )
        await client.connect(record, session.session_id)

        outcome = await client.slash_result("team", "viewerteam do the thing", [])

        assert (outcome.get("data") or {}).get("type") == "team_attached"
        assert (outcome.get("data") or {}).get("request") == "do the thing"
        assert session.active_team_name == "viewerteam", "the attach is the owner's job either way"

        # Give the runtime the same window the cell above needed to run a
        # turn. Nothing may appear in it — this is an absence assertion, so it
        # must be given real time to fail rather than being read immediately.
        transcript_path = directory / "transcript.jsonl"
        for _ in range(20):
            await asyncio.sleep(0.05)
            if transcript_path.exists() and "do the thing" in transcript_path.read_text(
                encoding="utf-8"
            ):
                raise AssertionError(
                    "the runtime admitted a request the client declared it would "
                    "submit itself; the user's command runs twice"
                )
    finally:
        if client is not None:
            client.close()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_published_record_carries_this_runtime_s_build(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The version channel, end to end: what a viewer reads before it dials.

    The record is the earliest hello there is — an attach client holds it
    before the socket is open — so the stamp has to be on the file a real
    ``registry.scan`` returns, not merely on an in-memory dataclass.
    """
    from local_operator.update import installed_version

    directory = headless_tui_env / "sessions" / "stampsess001"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, [])
    await server.start_in_process()
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        assert record.version == installed_version(), (
            "a runtime that cannot say what build it runs leaves every viewer "
            "unable to name the skew it is about to hit"
        )
    finally:
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_record_carries_the_source_ref_when_lop_update_recorded_one(
    headless_tui_env: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same-version rebuilds are this host's common drift, so the ref matters.

    ``lop-update`` writes ``<sha> <tag>`` into ``.lop-source`` at the install
    root. With a fake marker in place the runtime must publish that sha —
    without it, two builds of one release are indistinguishable on the wire
    and the drift goes unreported.
    """
    marker_root = tmp_path / "prefix"
    marker_root.mkdir()
    (marker_root / ".lop-source").write_text("feedfacecafe1234 v0.49.0\n", encoding="utf-8")

    import local_operator.update as update_mod

    # The REAL ``installed_build``, pointed at the fixture prefix: the marker
    # file is genuinely parsed, so a change that stopped reading ``.lop-source``
    # (or read the tag instead of the sha) fails here. Patching the function
    # wholesale would assert only that the server copies two attributes.
    monkeypatch.setattr(
        update_mod,
        "installed_build",
        lambda *_a, **_k: update_mod.BuildStamp(
            version=update_mod.installed_version(),
            source_ref=update_mod.source_ref(marker_root),
        ),
    )

    directory = headless_tui_env / "sessions" / "stampsess002"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, [])
    await server.start_in_process()
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        assert record.source_ref == "feedfacecafe1234"
    finally:
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_transient_viewer_drop_mid_turn_does_not_paint_interrupted(
    headless_tui_env: Path, workspace: Path
) -> None:
    """Test 23: a server-side drop mid-turn must not paint ``interrupted``.

    The runtime keeps working; the viewer re-binds to the same pid; the
    ledger never gains an aborted artefact; the turn then completes and
    paints the assistant row once.
    """
    from local_operator.harness.types import AgentTool, TextContent, ToolResult
    from local_operator.session.remote import RemoteSession
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.tool_card import ToolCard
    from tests.e2e.harness import (
        drain,
        tool_call_turn,
        transcript_text,
        wait_for_adoption,
    )

    started = asyncio.Event()
    released = asyncio.Event()

    async def execute_hang(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        started.set()
        await released.wait()
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="hang",
            content=[TextContent(text="hung done")],
        )

    hang = AgentTool(
        name="hang",
        parameters={"type": "object", "properties": {}},
        execute=execute_hang,
        interruptible=True,
    )
    directory = headless_tui_env / "sessions" / "dropmidturn01"
    directory.mkdir(parents=True)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Hanging now.",
                tool_name="hang",
                tool_call_id="e2e-hang-1",
                arguments={},
            ),
            text_turn("The hang finished normally."),
        ]
    )
    session = build_session(directory, stream, tools=[hang], cwd=workspace)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    import os

    (directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )
        pid_before = viewer.runtime_pid

        async def factory() -> RemoteSession:
            assert viewer is not None
            return viewer

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            owner_turn = asyncio.create_task(session.prompt("please hang"))
            for _ in range(400):
                await drain(pilot, cycles=2)
                if started.is_set() and list(app.query(ToolCard)):
                    break
                await asyncio.sleep(0.05)
            assert started.is_set(), "the hang tool never started on the runtime"
            painted = transcript_text(app)
            assert "interrupted" not in painted.lower()

            conns = [
                conn
                for conn in list(server._clients.values())
                if conn.wants_events and conn.wants_frontend
            ]
            assert conns, "no full-TUI client to drop"
            server._drop_client(conns[0], reason="test")

            rebound = False
            for _ in range(400):
                await drain(pilot, cycles=2)
                if (
                    viewer.runtime_pid == pid_before
                    and viewer.runtime_pid is not None
                    and not viewer._recovering
                    and viewer._client is not None
                    and viewer._client.connected
                ):
                    rebound = True
                    break
                await asyncio.sleep(0.05)
            painted = transcript_text(app)
            assert rebound, (
                f"viewer never re-bound to pid {pid_before}; "
                f"now pid={viewer.runtime_pid} recovering={viewer._recovering} "
                f"streaming={viewer.is_streaming}"
            )
            assert "interrupted" not in painted.lower(), painted
            assert "\u2298" not in painted, painted

            released.set()
            await asyncio.wait_for(owner_turn, timeout=15)
            for _ in range(200):
                await drain(pilot, cycles=2)
                painted = transcript_text(app)
                if "The hang finished normally." in painted:
                    break
                await asyncio.sleep(0.05)
            painted = transcript_text(app)
            assert "interrupted" not in painted.lower(), painted
            assert painted.count("The hang finished normally.") == 1, painted
            assert session.is_streaming is False
    finally:
        released.set()
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await handle.dispose()
