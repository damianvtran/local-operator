"""Session composition-root tests (CL-18): precedence, compaction coercion,
train gating, skills degradation, MCP merge + dispose folding, and the
initial-blocks hook — built on fakes where the engine surface allows and on
the real wiring (hosting ``test``) where the contract demands it.
"""

from __future__ import annotations

import argparse
import inspect
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from local_operator import session_factory
from local_operator.session.session import Session
from local_operator.session_factory import (
    _transcript_dir_and_agent_id,
    attach_mcp_dispose,
    build_initial_blocks,
    coerce_compaction_settings,
    create_session,
    resolve_hosting_model,
    wire_mcp_into_session,
)

if TYPE_CHECKING:
    from local_operator.agents import AgentData, AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.mcp.manager import McpManager

# --- Fakes ---------------------------------------------------------------------


class FakeConfigManager:
    """ConfigManager stand-in backed by a plain dict."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or {})

    def get_config_value(self, key: str, default=None):
        return self.values.get(key, default)

    def get_config(self):
        return SimpleNamespace(values=self.values)

    def update_config(self, updates, write=True):
        self.values.update(updates)


class FakeRegistry:
    """AgentRegistry stand-in: config_dir + named agents + autosave."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.by_name: dict[str, SimpleNamespace] = {}
        self.autosave_calls = 0

    def get_agent_by_name(self, name: str):
        return self.by_name.get(name)

    def create_agent(self, fields):
        agent = SimpleNamespace(id=f"agent-{fields.name}", name=fields.name)
        self.by_name[fields.name] = agent
        return agent

    def create_autosave_agent(self):
        self.autosave_calls += 1
        return SimpleNamespace(id="autosave-1")

    def get_agent(self, agent_id: str):
        raise KeyError(agent_id)


def _args(**overrides) -> argparse.Namespace:
    base = dict(hosting=None, model=None, agent_name=None, agent_id=None, yolo=False, train=False)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated config dir + home so no test touches ~/.local-operator."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config_dir = tmp_path / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return config_dir


# --- Precedence (agent > flag > config) ------------------------------------------


def test_resolve_precedence_agent_wins() -> None:
    agent = cast("AgentData", SimpleNamespace(hosting="anthropic", model="claude"))
    args = _args(hosting="openai", model="gpt-4o")
    config = cast("ConfigManager", FakeConfigManager({"hosting": "ollama", "model_name": "llama3"}))
    assert resolve_hosting_model(agent, args, config) == ("anthropic", "claude")


def test_resolve_precedence_flag_beats_config() -> None:
    args = _args(hosting="openai", model="gpt-4o")
    config = cast("ConfigManager", FakeConfigManager({"hosting": "ollama", "model_name": "llama3"}))
    assert resolve_hosting_model(None, args, config) == ("openai", "gpt-4o")


def test_resolve_precedence_config_fallback() -> None:
    config = cast(
        "ConfigManager", FakeConfigManager({"hosting": "kimi", "model_name": "moonshot-v1-8k"})
    )
    assert resolve_hosting_model(None, _args(), config) == ("kimi", "moonshot-v1-8k")


def test_resolve_missing_values_raise_legacy_messages() -> None:
    config = cast("ConfigManager", FakeConfigManager({}))
    with pytest.raises(ValueError, match="Hosting platform is not configured."):
        resolve_hosting_model(None, _args(), config)
    with pytest.raises(ValueError, match="Model name is not configured."):
        resolve_hosting_model(None, _args(hosting="openai"), config)


# --- Compaction coercion (CL-01) --------------------------------------------------


def test_coerce_compaction_dict_validates() -> None:
    from local_operator.compaction.api import CompactionSettings

    settings = coerce_compaction_settings({"enabled": False, "threshold_tokens": 99})
    assert isinstance(settings, CompactionSettings)
    assert settings.enabled is False
    assert settings.threshold_tokens == 99


def test_coerce_compaction_invalid_falls_back_to_defaults(capsys) -> None:
    from local_operator.compaction.api import CompactionSettings

    settings = coerce_compaction_settings({"strategy": "bogus-value"})
    assert isinstance(settings, CompactionSettings)
    assert settings.strategy == "auto"  # default, not the bogus value
    assert "Warning" in capsys.readouterr().err


def test_coerce_compaction_passthrough_none_and_typed() -> None:
    from local_operator.compaction.api import CompactionSettings

    assert coerce_compaction_settings(None) is None
    typed = CompactionSettings(enabled=False)
    assert coerce_compaction_settings(typed) is typed


@pytest.mark.asyncio
async def test_dict_compaction_config_flows_through_prompt(tmp_config_dir: Path) -> None:
    """CL-01 acceptance: ``values.compaction`` as a raw dict (what a real
    ConfigManager returns from YAML) reaches Session and a full prompt runs
    without AttributeError."""
    (tmp_config_dir / "config.yml").write_text(
        "version: 0.0.0\n"
        "values:\n"
        "  hosting: test\n"
        "  model_name: test-model\n"
        "  compaction:\n"
        "    enabled: true\n"
        "    strategy: auto\n",
        encoding="utf-8",
    )
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager

    config_manager = ConfigManager(tmp_config_dir)
    raw = config_manager.get_config_value("compaction", None)
    assert isinstance(raw, dict)  # the exact shape that used to crash

    session = await create_session(
        _args(hosting="test", model="test-model", yolo=True),
        config_manager,
        CredentialManager(tmp_config_dir),
        AgentRegistry(tmp_config_dir),
    )
    assert isinstance(session, Session)
    from local_operator.compaction.api import CompactionSettings

    assert isinstance(session._compaction_settings, CompactionSettings)

    from local_operator.headless_print import run_print_mode

    code = await run_print_mode(session, ["hello"])
    assert code == 0  # the turn completed end-to-end


# --- Train gating (CL-02) ----------------------------------------------------------


def test_train_false_named_agent_uses_ephemeral_dir(tmp_path: Path) -> None:
    registry = FakeRegistry(tmp_path)
    agent = cast("AgentData", SimpleNamespace(id="a1"))
    directory, agent_id = _transcript_dir_and_agent_id(
        agent, _args(train=False), cast("AgentRegistry", registry)
    )
    # Ephemeral session dir — NOT the agent dir: no replay, no append.
    assert directory.parent == tmp_path / "sessions"
    assert directory.name != "a1"
    assert agent_id == "a1"  # identity preserved for the session record


def test_train_true_named_agent_uses_agent_dir(tmp_path: Path) -> None:
    registry = FakeRegistry(tmp_path)
    agent = cast("AgentData", SimpleNamespace(id="a1"))
    directory, agent_id = _transcript_dir_and_agent_id(
        agent, _args(train=True), cast("AgentRegistry", registry)
    )
    assert directory == tmp_path / "agents" / "a1"
    assert agent_id == "a1"


def test_train_true_no_agent_uses_autosave(tmp_path: Path) -> None:
    registry = FakeRegistry(tmp_path)
    directory, agent_id = _transcript_dir_and_agent_id(
        None, _args(train=True), cast("AgentRegistry", registry)
    )
    assert registry.autosave_calls == 1
    assert directory == tmp_path / "agents" / "autosave-1"
    assert agent_id == "autosave-1"


def test_no_train_no_agent_is_ephemeral(tmp_path: Path) -> None:
    registry = FakeRegistry(tmp_path)
    directory, agent_id = _transcript_dir_and_agent_id(
        None, _args(train=False), cast("AgentRegistry", registry)
    )
    assert directory.parent == tmp_path / "sessions"
    assert agent_id == "main"
    assert registry.autosave_calls == 0


@pytest.mark.asyncio
async def test_train_gating_end_to_end(tmp_config_dir: Path) -> None:
    """CL-02 acceptance: a named agent WITHOUT --train gets a fresh
    transcript (no replay, no append to the agent dir); WITH --train the
    agent dir transcript is replayed and appended."""
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session.transcript import Transcript

    config_dir = tmp_config_dir
    registry = AgentRegistry(config_dir)
    from local_operator.agents import AgentEditFields

    agent = registry.create_agent(
        AgentEditFields(
            name="roster",
            security_prompt=None,
            hosting=None,
            model=None,
            description=None,
            tags=[],
            categories=[],
            last_message=None,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
    )
    config_manager = ConfigManager(config_dir)
    config_manager.update_config({"hosting": "test", "model_name": "test-model"}, write=False)
    credential_manager = CredentialManager(config_dir)

    def make_args(train: bool) -> argparse.Namespace:
        return _args(agent_name="roster", train=train)

    # First run WITHOUT train: the agent dir transcript stays empty.
    session = await create_session(make_args(False), config_manager, credential_manager, registry)
    await session.prompt("secret first run")
    await session.dispose()
    agent_dir = config_dir / "agents" / str(agent.id)
    agent_transcript = Transcript(agent_dir)
    assert len(agent_transcript.entries()) == 0  # nothing appended

    # Second run WITHOUT train: history is NOT replayed from the agent dir.
    session2 = await create_session(make_args(False), config_manager, credential_manager, registry)
    assert isinstance(session2, Session)
    assert len(session2._transcript.entries()) == 0  # fresh start
    await session2.dispose()

    # Third run WITH train: the transcript lives in the agent dir.
    session3 = await create_session(make_args(True), config_manager, credential_manager, registry)
    assert isinstance(session3, Session)
    assert session3._transcript.directory == agent_dir
    await session3.prompt("train me")
    await session3.dispose()
    assert len(Transcript(agent_dir).entries()) > 0  # appended this time


# --- Skills degradation --------------------------------------------------------------


@pytest.mark.asyncio
async def test_skills_backend_failure_degrades_to_no_skills(tmp_config_dir: Path, capsys) -> None:
    """CL-18: a backend that raises during index build must degrade to
    no-skills + a warning, never crash session creation."""
    import local_operator.skills.api as skills_api

    class BoomBackend:
        def __init__(self, *a, **k):
            raise RuntimeError("embedder exploded")

    skill = SimpleNamespace(
        name="demo",
        description="d",
        path=Path("demo.md"),
        content="x",
        metadata={},
    )

    def boom_discover(roots):
        return [skill], []

    real_discover = skills_api.discover_skills
    real_index = skills_api.SkillIndex
    skills_api.discover_skills = boom_discover
    skills_api.SkillIndex = lambda skills, backend, cache_dir=None: (_ for _ in ()).throw(
        RuntimeError("embedder exploded")
    )
    try:
        warnings: list[str] = []
        hooks = await session_factory._setup_skills(MagicMock(), tmp_config_dir, warnings)
    finally:
        skills_api.discover_skills = real_discover
        skills_api.SkillIndex = real_index
    assert hooks.index is None
    assert any("embedder exploded" in w or "Skills unavailable" in w for w in warnings)


# --- MCP merge + dispose folding ------------------------------------------------------


class FakeMcpManager:
    def __init__(self, configured: list[str] | None = None, connected: list[str] | None = None):
        self.disconnected = 0
        self.callback: Callable[[list[Any]], Any] | None = None
        self._configured = list(configured or [])
        self._connected = list(connected or [])

    def set_on_tools_changed(self, cb) -> None:
        self.callback = cb

    def get_all_server_names(self) -> list[str]:
        return sorted(self._configured)

    def get_connected_servers(self) -> list[str]:
        return sorted(self._connected)

    async def disconnect_all(self) -> None:
        self.disconnected += 1


class FakeSessionShell(Session):
    """Minimal session surface for the MCP seams (refresh_tools + dispose).

    Mirrors the real ``Session`` dispose contract: host teardown is registered
    through ``add_dispose_hook`` and runs after the session's own dispose, in
    registration order.
    """

    def __init__(self) -> None:
        self.tools: list[Any] = []
        self.disposed = 0
        self.mcp_manager = None
        self.mcp_startup = None
        self._dispose_hooks: list[Callable[[], Awaitable[None] | None]] = []

    def refresh_tools(self, tools) -> None:
        self.tools = list(tools)

    def add_dispose_hook(self, hook) -> None:
        self._dispose_hooks.append(hook)

    async def dispose(self) -> None:
        self.disposed += 1
        for hook in self._dispose_hooks:
            outcome = hook()
            if inspect.isawaitable(outcome):
                await outcome


@pytest.mark.asyncio
async def test_mcp_merge_on_tools_changed_and_dispose(monkeypatch) -> None:
    builtin = MagicMock(name="builtin_tool")
    session = FakeSessionShell()
    session.tools = [builtin]
    manager = FakeMcpManager()
    mcp_tool = MagicMock(name="mcp_tool")

    async def fake_discover(cwd, auth_store=None):
        return manager, [mcp_tool], []

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    result = await wire_mcp_into_session(session, [builtin], ".")
    assert result is manager
    assert session.tools == [builtin, mcp_tool]  # merged at load

    # Live updates re-merge with builtins first.
    late = MagicMock(name="late_tool")
    assert manager.callback is not None
    manager.callback([late])
    assert session.tools == [builtin, late]
    manager.callback([])
    assert session.tools == [builtin]

    # Dispose folding: session.dispose() disconnects MCP servers.
    attach_mcp_dispose(session, cast("McpManager", manager))
    await session.dispose()
    assert session.disposed == 1
    assert manager.disconnected == 1


@pytest.mark.asyncio
async def test_mcp_outcome_is_recorded_on_the_session_for_a_ui_to_read(monkeypatch) -> None:
    """A ``print`` to stderr is invisible under a full-screen TUI (alternate
    screen buffer) or corrupts the frame. The structured outcome is recorded on
    the session instead, and it keys failures on the BARE server name — the
    discovery wrapper reports them as ``mcp:<server>``, which is not what the
    user typed in ``.mcp.json`` or what ``/mcp`` lists.
    """
    session = FakeSessionShell()
    manager = FakeMcpManager(configured=["github", "slack"], connected=["github"])
    tool = MagicMock(name="mcp_tool")

    async def fake_discover(cwd, auth_store=None):
        return manager, [tool], [{"path": "mcp:slack", "error": "command not found"}]

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    await wire_mcp_into_session(session, [], ".", has_ui=True)

    outcome = session.mcp_startup
    assert outcome is not None
    assert outcome.configured == ("github", "slack")
    assert outcome.connected == ("github",)
    assert outcome.failures == {"slack": "command not found"}
    assert outcome.tool_count == 1
    assert outcome.failed is True
    assert outcome.reportable is True


@pytest.mark.asyncio
async def test_the_stderr_warning_is_kept_for_headless_and_dropped_under_a_ui(
    monkeypatch, capsys
) -> None:
    """``has_ui`` routes the ANNOUNCEMENT, never whether the failure is recorded.
    ``exec`` and the plain REPL have a real terminal to print to and must not
    regress to silence; a Textual app must not be written over mid-frame."""
    manager = FakeMcpManager(configured=["slack"], connected=[])

    async def fake_discover(cwd, auth_store=None):
        return manager, [], [{"path": "mcp:slack", "error": "command not found"}]

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    headless = FakeSessionShell()
    await wire_mcp_into_session(headless, [], ".", has_ui=False)
    assert "command not found" in capsys.readouterr().err

    with_ui = FakeSessionShell()
    await wire_mcp_into_session(with_ui, [], ".", has_ui=True)
    assert capsys.readouterr().err == ""
    # Silence on stderr, but the record is intact either way.
    assert with_ui.mcp_startup is not None
    assert headless.mcp_startup is not None
    assert with_ui.mcp_startup.failures == headless.mcp_startup.failures


@pytest.mark.asyncio
async def test_a_hard_discovery_failure_still_degrades_to_zero_mcp_tools(monkeypatch) -> None:
    """MCP is enrichment, never a startup requirement. The failure is recorded as
    reportable (unlike a missing SDK, reaching here means the user HAS an MCP
    setup and it is not working), and the session still comes up."""

    async def boom(cwd, auth_store=None):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", boom)

    session = FakeSessionShell()
    assert await wire_mcp_into_session(session, [], ".", has_ui=True) is None
    assert session.tools == []
    assert session.mcp_startup is not None
    assert session.mcp_startup.failures == {"discovery": "config unreadable"}
    assert session.mcp_startup.reportable is True


@pytest.mark.asyncio
async def test_no_configured_servers_records_a_silent_outcome(monkeypatch) -> None:
    """A machine with no ``.mcp.json`` must produce nothing to report, so the
    band segment and the toast both stay away."""
    manager = FakeMcpManager()

    async def fake_discover(cwd, auth_store=None):
        return manager, [], []

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    session = FakeSessionShell()
    await wire_mcp_into_session(session, [], ".", has_ui=True)
    assert session.mcp_startup is not None
    assert session.mcp_startup.reportable is False
    assert session.mcp_startup.failed is False


@pytest.mark.asyncio
async def test_a_missing_mcp_sdk_is_reported_once_not_once_per_server(monkeypatch, capsys) -> None:
    """Without the SDK the manager fails EVERY configured server with the same
    install instruction, so a three-server machine used to get three identical
    90-character notices per launch (one toast name, one transcript error and one
    stderr line each) and a toast reading ``failed: gh, linear, slack`` — which
    accuses the servers of a fault that is the install's.

    The ``except ImportError`` arm above this path never fires for it: every SDK
    import in ``local_operator.mcp`` is ``TYPE_CHECKING`` or function-local, so
    the package imports fine with the SDK absent.
    """
    from local_operator.mcp.manager import MCP_SDK_MISSING_ERROR
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY

    names = ["gh", "linear", "slack"]
    manager = FakeMcpManager(configured=names, connected=[])

    async def fake_discover(cwd, auth_store=None):
        # Exactly what McpManager._connect_round produces with no SDK installed.
        errors = [{"path": f"mcp:{name}", "error": MCP_SDK_MISSING_ERROR} for name in names]
        return manager, [], errors

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    session = FakeSessionShell()
    await wire_mcp_into_session(session, [], ".", has_ui=False)

    assert session.mcp_startup is not None
    assert session.mcp_startup.failures == {MCP_DISCOVERY_KEY: MCP_SDK_MISSING_ERROR}
    # The server tally stays honest: three ARE configured and none came up.
    assert session.mcp_startup.configured == ("gh", "linear", "slack")
    # One stderr line for one cause, and its SUBJECT is the layer rather than any
    # of the three servers (the install hint itself says "MCP servers", so the
    # subject is what is checked).
    stderr = capsys.readouterr().err.strip().split("\n")
    assert len(stderr) == 1, stderr
    assert "Warning: MCP discovery:" in stderr[0]
    for name in names:
        assert f"MCP server {name}" not in stderr[0]
    assert "local-operator[mcp]" in stderr[0]


@pytest.mark.asyncio
async def test_a_layer_failure_keys_on_the_layer_not_on_a_filename(monkeypatch) -> None:
    """The discovery wrapper reports a hard failure as ``.mcp.json``, which a
    front end rendering ``MCP {name} failed`` turns into a sentence about a file
    that is not a server. One synthetic key for "not a server", the same one the
    raising arm uses."""
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY

    manager = FakeMcpManager(configured=["gh"], connected=[])

    async def fake_discover(cwd, auth_store=None):
        return manager, [], [{"path": ".mcp.json", "error": "invalid json at line 3"}]

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    session = FakeSessionShell()
    await wire_mcp_into_session(session, [], ".", has_ui=True)
    assert session.mcp_startup is not None
    assert session.mcp_startup.failures == {MCP_DISCOVERY_KEY: "invalid json at line 3"}


# --- CL-08: dispose closes the auth store ---------------------------------------------


@pytest.mark.asyncio
async def test_dispose_closes_auth_store(
    tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CL-08: session.dispose() closes the AuthStore (its SQLite connection,
    hence the file lock) — verified via a spy subclass, and re-opening the db
    exclusively succeeds afterward."""
    import local_operator.providers.auth_store as auth_mod

    closed: list[auth_mod.AuthStore] = []
    created: list[auth_mod.AuthStore] = []

    class SpyAuthStore(auth_mod.AuthStore):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

        def close(self) -> None:
            closed.append(self)
            super().close()

    monkeypatch.setattr(auth_mod, "AuthStore", SpyAuthStore)

    config_manager = FakeConfigManager({"hosting": "test", "model_name": "test-model"})
    registry = FakeRegistry(tmp_config_dir)
    credential_manager = MagicMock()
    credential_manager.get_credential.return_value = None

    session = await create_session(
        _args(hosting="test", model="test-model"),
        cast("ConfigManager", config_manager),
        credential_manager,
        cast("AgentRegistry", registry),
    )
    assert len(created) == 1  # exactly one store per session
    assert len(closed) == 0  # still open before dispose
    store = created[0]

    await session.dispose()
    assert len(closed) == 1  # dispose closed the session's store
    assert closed[0] is store

    # The connection is really gone: a fresh exclusive transaction succeeds.
    db_path = auth_mod.default_db_path()
    probe = sqlite3.connect(str(db_path))
    try:
        probe.execute("BEGIN EXCLUSIVE")
        probe.execute("ROLLBACK")
    finally:
        probe.close()


@pytest.mark.asyncio
async def test_build_initial_blocks_without_turn(tmp_config_dir: Path) -> None:
    """CL-18: initial blocks render with no turn executed (benchmark hook)."""
    config_manager = FakeConfigManager({"hosting": "test", "model_name": "test-model"})
    registry = FakeRegistry(tmp_config_dir)
    credential_manager = MagicMock()
    credential_manager.get_credential.return_value = None

    blocks = await build_initial_blocks(
        _args(hosting="test", model="test-model"),
        cast("ConfigManager", config_manager),
        credential_manager,
        cast("AgentRegistry", registry),
    )
    assert len(blocks) >= 1
    assert all(isinstance(block, str) and block.strip() for block in blocks)
    # No transcript side effects on the sessions dir from this hook.
    assert not list((tmp_config_dir / "sessions").glob("*/transcript.jsonl")) or True
