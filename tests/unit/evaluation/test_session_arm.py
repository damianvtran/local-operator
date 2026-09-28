"""The session arm: declaration, bridge discipline, the MCP wire, the surface.

These tests are the PR's falsifiable core, and each maps to a claim the design
makes:

* **The action surface is a session-scoped MCP server** (design §6). The wire
  test drives the REAL stack end to end: a real ``McpManager`` spawns the real
  ``local_operator.evaluation.action_server`` child, which forwards over a
  UNIX socket to a live ``ActionBridge``, which executes and renders -- and the
  ToolResult the model would see carries the rendered step text AND the frame
  bytes as an image block.
* **cwd and the MCP config stay inside the episode scratch** (PR 1's QA MUST,
  design §5). ``assert_declaration_resolved`` refuses a cwd or a discovered
  source outside the scratch root, by resolved path.
* **One batch per observation, the turn boundary re-arms** -- the token
  discipline is the loop-driven tool's, imported not re-stated.
* **The episode session carries the shipped tool surface**: opened through
  ``sdk.open_session`` with the action server declared, the session's live
  inventory holds the harness's own tools (``task``/``hub``/``team`` among
  them) alongside the minted action tool.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator import agent_shell, sdk
from local_operator.compaction.png import encode_grayscale_png
from local_operator.evaluation.action_server import (
    WIRE_READ_LIMIT_BYTES,
    decode_response,
    encode_call,
)
from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import (
    ExecuteResult,
    ExecutionReceipt,
    observation_content_id,
)
from local_operator.evaluation.protocol import (
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)
from local_operator.evaluation.session_arm import (
    ActionBridge,
    ObservationRenderer,
    SessionArmError,
    assert_declaration_resolved,
    declare_action_server,
    session_tool_names,
)
from local_operator.harness.types import (
    ImageContent,
    TextContent,
    ToolContext,
    TurnEndEvent,
)
from local_operator.mcp.manager import McpManager
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec


def _roots_free() -> SessionRoots:
    """Roots for tests that never touch disk or a session factory."""

    return SessionRoots(
        config_dir=Path("/nonexistent/pr2-config"),
        agent_home=Path("/nonexistent/pr2-home"),
        cwd=Path("/nonexistent/pr2-work"),
        allow_volatile=True,
    )


class _AsyncContext:
    async def __aenter__(self) -> Any:  # pragma: no cover - never entered
        raise AssertionError("closed before entering")

    async def __aexit__(self, *exc: Any) -> None:  # pragma: no cover
        return None


#: A REAL image: the frame reader validates media (`verify_artifact`), so a
#: fixture of arbitrary bytes would be refused by the same check the runner
#: uses -- which is the property the wire test needs to exercise.
PNG = encode_grayscale_png(1, 1, b"\x00")

DIGEST = "0" * 64


def _observation(
    *,
    sequence: int = 0,
    text: str = "screen A",
    frames: tuple[FrameRef, ...] = (),
    episode_id: str = "ep-session-arm",
) -> Observation:
    draft = Observation(
        task_id="task_001",
        episode_id=episode_id,
        sequence=sequence,
        observation_id="draft",
        text=text,
        frames=frames,
    )
    return draft.model_copy(update={"observation_id": observation_content_id(draft)})


def _frame(artifact_root: Path, data: bytes) -> FrameRef:
    digest = hashlib.sha256(data).hexdigest()
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / digest).write_bytes(data)
    size = FrameSize(width=800, height=600)
    return FrameRef(
        frame_id="frame-0",
        artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=len(data)),
        geometry=FrameGeometry(native=size, model_visible=size),
    )


def _result(*, input_observation: Observation, output_observation: Observation) -> ExecuteResult:
    return ExecuteResult(
        receipt=ExecutionReceipt(
            operation_id="op-test",
            action_batch_id=DIGEST,
            input_observation_id=input_observation.observation_id,
            output_observation_id=output_observation.observation_id,
            sequence=output_observation.sequence,
        ),
        observation=output_observation,
    )


def _bridge(
    tmp_path: Path,
    *,
    observation: Observation | None = None,
    execute: Any = None,
    max_steps: int = 8,
) -> ActionBridge:
    obs0 = observation or _observation()
    obs1 = _observation(sequence=1, text="screen B")

    async def default_execute(batch: Any) -> ExecuteResult:
        del batch
        return _result(input_observation=obs0, output_observation=obs1)

    bridge = ActionBridge(
        endpoint=tmp_path / "action-bridge.sock",
        surface=ActionSurface(),
        render=lambda observation: [TextContent(text=f"seen {observation.sequence}")],
        execute=execute or default_execute,
        max_steps=max_steps,
    )
    bridge.arm(obs0)
    return bridge


class TestDeclaration:
    def test_writes_the_server_entry_and_mints_the_name(self, tmp_path: Path) -> None:
        config = tmp_path / "home" / ".local-operator"
        config.mkdir(parents=True)
        work = tmp_path / "home" / "work"
        work.mkdir()
        decl = declare_action_server(
            config_dir=config,
            endpoint=tmp_path / "b.sock",
            surface=ActionSurface(paste_text=True),
            python_executable="/venv/python",
            cwd=work,
        )
        assert decl.tool_name == "mcp__episode_actions_apply_actions"
        document = json.loads((config / "mcp.json").read_text(encoding="utf-8"))
        entry = document["mcpServers"]["episode-actions"]
        assert entry["type"] == "stdio"
        assert entry["command"] == "/venv/python"
        assert entry["args"][:2] == ["-m", "local_operator.evaluation.action_server"]
        assert entry["args"][2] == "--endpoint"
        assert entry["args"][3] == str(tmp_path / "b.sock")
        assert entry["preloadTools"] is True
        assert entry["enabledTools"] == ["apply_actions"]
        assert entry["ownTurnOnly"] is True
        assert entry["cwd"] == str(work)

    def test_merges_an_existing_file(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        config.mkdir()
        (config / "mcp.json").write_text(
            json.dumps({"mcpServers": {"other": {"type": "stdio", "command": "true"}}}),
            encoding="utf-8",
        )
        declare_action_server(
            config_dir=config, endpoint=tmp_path / "b.sock", surface=ActionSurface()
        )
        document = json.loads((config / "mcp.json").read_text(encoding="utf-8"))
        assert set(document["mcpServers"]) == {"other", "episode-actions"}
        assert document["mcpServers"]["other"]["command"] == "true"

    def test_unreadable_existing_file_is_refused(self, tmp_path: Path) -> None:
        config = tmp_path / "config"
        config.mkdir()
        (config / "mcp.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SessionArmError, match="not readable JSON"):
            declare_action_server(
                config_dir=config, endpoint=tmp_path / "b.sock", surface=ActionSurface()
            )


class TestResolution:
    """The recorded MUST: the episode's MCP graph resolves inside its scratch."""

    def _declare(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        home = tmp_path / "home"
        config = home / ".local-operator"
        work = home / "work"
        config.mkdir(parents=True)
        work.mkdir()
        declare_action_server(
            config_dir=config, endpoint=tmp_path / "b.sock", surface=ActionSurface()
        )
        return home, config, work

    def test_sources_inside_scratch_are_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, config, work = self._declare(tmp_path)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
        sources = assert_declaration_resolved(cwd=work, config_dir=config, scratch_root=home)
        assert sources["episode-actions"] == str(config / "mcp.json")
        assert all(Path(source).resolve().is_relative_to(home) for source in sources.values())

    def test_a_cwd_outside_scratch_is_refused(self, tmp_path: Path) -> None:
        home, config, work = self._declare(tmp_path)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(SessionArmError, match="cwd"):
            assert_declaration_resolved(cwd=outside, config_dir=config, scratch_root=home)

    def test_a_config_outside_scratch_is_refused(self, tmp_path: Path) -> None:
        home, config, work = self._declare(tmp_path)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(SessionArmError, match="config dir"):
            assert_declaration_resolved(cwd=work, config_dir=outside, scratch_root=home)

    def test_a_dropped_declaration_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home, config, work = self._declare(tmp_path)
        (config / "mcp.json").unlink()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
        with pytest.raises(SessionArmError, match="was not discovered"):
            assert_declaration_resolved(cwd=work, config_dir=config, scratch_root=home)


class TestBridgeDiscipline:
    @pytest.mark.asyncio
    async def test_one_batch_per_observation_until_the_turn_ends(self, tmp_path: Path) -> None:
        executed: list[Any] = []
        bridge = _bridge(tmp_path, execute=None)
        original = bridge.execute

        async def record(batch: Any) -> Any:
            executed.append(batch)
            return await original(batch)

        bridge.execute = record
        wait = {"actions": [{"kind": "wait", "duration_ms": 50}]}

        first = await bridge.call(wait)
        assert first["is_error"] is False
        assert bridge.steps == 1
        assert len(executed) == 1

        second = await bridge.call(wait)
        assert second["is_error"] is True
        assert second["details"]["rejection_class"] == "second-batch"
        assert len(executed) == 1

        bridge.fold(TurnEndEvent())
        third = await bridge.call(wait)
        assert third["is_error"] is False
        assert len(executed) == 2

    @pytest.mark.asyncio
    async def test_a_finish_batch_is_terminal_and_never_executes(self, tmp_path: Path) -> None:
        executed: list[Any] = []
        bridge = _bridge(tmp_path, execute=None)
        original = bridge.execute

        async def record(batch: Any) -> Any:
            executed.append(batch)
            return await original(batch)

        bridge.execute = record
        reply = await bridge.call(
            {"actions": [{"kind": "finish", "status": "done", "reason": "complete"}]}
        )
        assert reply["is_error"] is False
        assert reply["details"]["terminal"] == "finish"
        assert bridge.terminal is True
        assert bridge.end_requested == "finish"
        assert executed == []

        after = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert after["is_error"] is True
        assert bridge.steps == 0

    @pytest.mark.asyncio
    async def test_the_step_budget_ends_the_episode_without_executing(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path, max_steps=1)
        wait = {"actions": [{"kind": "wait", "duration_ms": 50}]}
        assert (await bridge.call(wait))["is_error"] is False
        # The budget is checked at the TOP of a turn, exactly as the runner
        # checks it before the next decision -- so the refusal lands on the
        # first call after the turn boundary re-arms the token, and the
        # budgeted batch never runs.
        bridge.fold(TurnEndEvent())
        refusal = await bridge.call(wait)
        assert refusal["is_error"] is False
        assert refusal["details"]["terminal"] == "max-steps"
        assert bridge.end_requested == "max-steps"
        assert bridge.steps == 1

    @pytest.mark.asyncio
    async def test_an_unparseable_batch_is_refused_with_a_class(self, tmp_path: Path) -> None:
        bridge = _bridge(tmp_path)
        reply = await bridge.call({"actions": [{"kind": "warp", "x": 1, "y": 2}]})
        assert reply["is_error"] is True
        assert "rejection_class" in reply["details"]

    @pytest.mark.asyncio
    async def test_a_missing_pending_observation_is_refused(self, tmp_path: Path) -> None:
        bridge = ActionBridge(
            endpoint=tmp_path / "b.sock",
            surface=ActionSurface(),
            render=lambda observation: [TextContent(text="x")],
            execute=None,  # type: ignore[arg-type]
            max_steps=2,
        )
        reply = await bridge.call({"actions": [{"kind": "wait", "duration_ms": 50}]})
        assert reply["is_error"] is True


class TestMcpWire:
    @pytest.mark.asyncio
    async def test_a_real_server_delivers_the_rendered_frame(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        work = home / "work"
        config = home / ".local-operator"
        artifacts = tmp_path / "artifacts"
        for path in (work, config, artifacts):
            path.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

        obs0 = _observation(frames=(_frame(artifacts, PNG),))
        obs1 = _observation(sequence=1, text="screen B", frames=(_frame(artifacts, PNG),))
        executed: list[Any] = []

        async def execute(batch: Any) -> ExecuteResult:
            executed.append(batch)
            return _result(input_observation=obs0, output_observation=obs1)

        endpoint = tmp_path / "b.sock"
        renderer = ObservationRenderer(artifact_root=artifacts)
        bridge = ActionBridge(
            endpoint=endpoint,
            surface=ActionSurface(),
            render=renderer.render,
            execute=execute,
            max_steps=5,
        )
        bridge.arm(obs0)
        await bridge.start()
        try:
            decl = declare_action_server(
                config_dir=config,
                endpoint=endpoint,
                surface=ActionSurface(),
                python_executable=sys.executable,
                cwd=work,
            )
            manager = McpManager(str(work))
            try:
                await manager.discover_and_connect()
                await manager.wait_settled(20.0)
                tools = {tool.name for tool in manager.get_tools()}
                assert decl.tool_name in tools, tools
                tool = next(tool for tool in manager.get_tools() if tool.name == decl.tool_name)
                result = await tool.execute(
                    "c1",
                    {"actions": [{"kind": "click", "frame_id": "frame-0", "x": 3, "y": 4}]},
                    None,
                    None,
                    ToolContext(),
                )
            finally:
                await manager.disconnect_all()
        finally:
            await bridge.stop()

        assert result.is_error is False
        assert len(executed) == 1
        texts = [block.text for block in result.content if isinstance(block, TextContent)]
        images = [block for block in result.content if isinstance(block, ImageContent)]
        assert any("Step: 1" in text for text in texts), texts
        assert len(images) == 1
        assert base64.b64decode(images[0].data) == PNG

    @pytest.mark.asyncio
    async def test_a_dead_bridge_surfaces_as_an_error_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        work = home / "work"
        config = home / ".local-operator"
        for path in (work, config):
            path.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
        decl = declare_action_server(
            config_dir=config,
            endpoint=tmp_path / "missing.sock",
            surface=ActionSurface(),
            python_executable=sys.executable,
            cwd=work,
        )
        manager = McpManager(str(work))
        try:
            await manager.discover_and_connect()
            await manager.wait_settled(20.0)
            tool = next(tool for tool in manager.get_tools() if tool.name == decl.tool_name)
            result = await tool.execute(
                "c1", {"actions": [{"kind": "wait", "duration_ms": 50}]}, None, None, ToolContext()
            )
        finally:
            await manager.disconnect_all()
        assert result.is_error is True
        assert "bridge" in result.text


class TestEpisodeSessionSurface:
    @pytest.fixture
    def scratch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionRoots:
        home = tmp_path / "home"
        root = home / ".local-operator"
        agent_home = home / "local-operator-home"
        work = home / "work"
        for path in (root, agent_home, work):
            path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(agent_home))
        monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
        monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
        monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
        return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)

    @pytest.mark.asyncio
    async def test_the_inventory_holds_the_shipped_tools_and_the_action_tool(
        self, scratch: SessionRoots, tmp_path: Path
    ) -> None:
        bridge = ActionBridge(
            endpoint=tmp_path / "b.sock",
            surface=ActionSurface(),
            render=lambda observation: [TextContent(text="x")],
            execute=None,  # type: ignore[arg-type]
            max_steps=2,
        )
        bridge.arm(_observation())
        await bridge.start()
        try:
            decl = declare_action_server(
                config_dir=scratch.config_path,
                endpoint=tmp_path / "b.sock",
                surface=ActionSurface(),
                python_executable=sys.executable,
                cwd=scratch.cwd_path,
            )
            assert_declaration_resolved(
                cwd=scratch.cwd_path,
                config_dir=scratch.config_path,
                scratch_root=scratch.cwd_path.parent,
            )
            spec = SessionSpec(
                hosting="test", model="mock", approvals=ApprovalPolicy.auto(), name="arm-surface"
            )
            async with sdk.open_session(spec, roots=scratch) as session:
                deadline = time.monotonic() + 25.0
                names: tuple[str, ...] = ()
                while time.monotonic() < deadline:
                    names = session_tool_names(session)
                    if decl.tool_name in names:
                        break
                    await asyncio.sleep(0.1)
                assert decl.tool_name in names, names
                assert {"task", "team", "hub", "jobs"} <= set(names), names
                assert len(names) >= 20
        finally:
            await bridge.stop()


class TestActionToolGate:
    """The settle gate: prompt only once the action tool provably exists.

    MCP discovery is asynchronous by design -- the action server is spawned and
    indexed after ``open`` -- and the first request's tool array is published
    ONCE per turn. A prompt that wins that race answers every call with
    "Tool not found: <action tool>" (measured on the pilot), so the arm holds
    the prompt until the tool is in the LIVE inventory and refuses to run when
    it never arrives.
    """

    @pytest.mark.asyncio
    async def test_the_gate_holds_until_the_tool_is_live(self, tmp_path: Path) -> None:
        from local_operator.evaluation.session_arm import (
            EpisodeSession,
            _await_action_tool,
        )

        class _Session:
            def __init__(self) -> None:
                self._tools: list[Any] = [SimpleNamespace(name="bash")]
                self.mcp_startup = None
                self.mcp_manager = None

        class _Declaration:
            tool_name = "mcp__episode_actions_apply_actions"

        session = _Session()
        handle = EpisodeSession(session=session, roots=_roots_free(), _context=_AsyncContext())
        record = cast(Any, SimpleNamespace(write=lambda *a, **k: None))
        declaration = cast(Any, _Declaration())
        assert await _await_action_tool(handle, declaration, record) is False

        # The tool arriving in the LIVE inventory -- the same list the first
        # request publishes from -- flips the gate without a reopen.
        session._tools.append(SimpleNamespace(name=_Declaration.tool_name))
        assert await _await_action_tool(handle, declaration, record) is True

    @pytest.mark.asyncio
    async def test_tool_names_are_live_not_frozen(self) -> None:
        from local_operator.evaluation.session_arm import EpisodeSession

        class _Session:
            def __init__(self) -> None:
                self._tools: list[Any] = []

        session = _Session()
        handle = EpisodeSession(session=session, roots=_roots_free(), _context=_AsyncContext())
        assert handle.tool_names == ()
        session._tools.append(SimpleNamespace(name="bash"))
        assert handle.tool_names == ("bash",)

    @pytest.mark.asyncio
    async def test_an_over_long_socket_path_is_refused_before_binding(self, tmp_path: Path) -> None:
        # sockaddr_un is ~104 bytes on macOS; a scratch deep enough to trip it
        # must fail here, in words, rather than as a connect error in the child.
        long_dir = tmp_path / ("d" * 60) / ("e" * 60)
        long_dir.mkdir(parents=True)
        bridge = ActionBridge(
            endpoint=long_dir / "b.sock",
            surface=ActionSurface(),
            render=lambda observation: [TextContent(text="x")],
            execute=None,  # type: ignore[arg-type]
            max_steps=1,
        )
        with pytest.raises(SessionArmError, match="too long for a UNIX socket"):
            await bridge.start()

    def test_cleanup_without_receipts_forces_rescue(self, tmp_path: Path) -> None:
        from local_operator.evaluation.session_arm import _cleanup_forces_rescue

        # No receipts is the dead-worker case: "attempted" is never evidence.
        assert _cleanup_forces_rescue(None, ()) is True  # type: ignore[arg-type]


class TestWireReadLimits:
    """Both ends of the socket must carry the wire's read limit, not the default.

    Added after the 2026-09-28 paid probe: a 476 KiB observation frame made
    ``forward_call``'s read raise "Separator is not found, and chunk exceed the
    limit" (asyncio's 64 KiB default), so every EXECUTED batch was answered to
    the model as unreachable while the desktop had already acted.
    """

    @pytest.mark.asyncio
    async def test_a_call_frame_larger_than_the_stream_default_is_served(
        self, tmp_path: Path
    ) -> None:
        # The request direction carries the same limit: a paste action may
        # legally carry ``max_type_chars`` characters in one frame, larger
        # than the default read limit.
        surface = ActionSurface(paste_text=True, max_type_chars=100_000)
        executed: list[Any] = []
        obs_in = _observation()
        obs_out = _observation(sequence=1, text="screen B")

        async def execute(batch: Any) -> ExecuteResult:
            executed.append(batch)
            return _result(input_observation=obs_in, output_observation=obs_out)

        bridge = ActionBridge(
            endpoint=tmp_path / "big-call.sock",
            surface=surface,
            render=lambda observation: [TextContent(text="seen")],
            execute=execute,
            max_steps=3,
        )
        bridge.arm(obs_in)
        await bridge.start()
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.open_unix_connection(
                str(bridge.endpoint), limit=WIRE_READ_LIMIT_BYTES
            )
            writer.write(
                encode_call(
                    {
                        "actions": [
                            {
                                "kind": "paste_text",
                                "keys": ["META", "v"],
                                "clipboard_policy": "overwrite",
                                "text": "x" * 90_000,
                            }
                        ]
                    }
                )
            )
            await writer.drain()
            reply = decode_response(await reader.readline())
        finally:
            if writer is not None:
                writer.close()
            await bridge.stop()
        assert reply.get("is_error") is False
        assert bridge.steps == 1
        assert len(executed) == 1
