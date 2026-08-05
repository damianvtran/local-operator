"""Session composition root for the CLI-facing entry points.

Why this module exists: ``cli.py`` (interactive + exec), ``exec_worker.py``
(detached background runs) and later the server facade all need the SAME
wiring from parsed args plus the three legacy managers to a harness
:class:`~local_operator.session.protocol.SessionProtocol`. Centralising the
wiring here keeps the precedence rules, the transcript-directory policy, the
skills integration, and the lazy-import discipline in exactly one place.

Constraints honored here (docs/REWRITE.md):

- No module-level imports of providers / session internals / skills / TUI.
  Every engine import happens inside functions, so importing this module is
  cheap and stays valid while parallel rewrite streams are mid-flight.
- Hosting/model resolution precedence: **agent > CLI flag > config file**
  (the legacy bootstrap order, minus the server-only request overrides).
- Skills are wired end-to-end (orchestrator integration duty): discovery +
  index build at session creation, per-turn semantic selection inside the
  system-blocks provider, and the ``skill://`` resolver adapter handed to
  both the tools context and the session. ANY skills failure degrades to
  "no skills" with a warning — never a crashed startup.

``create_session`` is async: the TUI's committed factory contract is
``Callable[[], Awaitable[SessionProtocol]]`` and the eager skill-index build
needs an await. Headless callers wrap it in ``asyncio.run``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from local_operator.harness.types import AgentMessage, Message, TextContent

if TYPE_CHECKING:
    from local_operator.session.protocol import SessionProtocol


def coerce_compaction_settings(raw: Any) -> Any:
    """Coerce ``values.compaction`` into a :class:`CompactionSettings` (CL-01).

    ``ConfigManager`` returns the YAML shape verbatim — a plain ``dict`` — but
    the session consumes attribute-style settings. ``None`` and already-typed
    settings pass through; a dict is validated; an invalid dict degrades to
    defaults with a warning (a bad compaction block must never block startup).
    """
    if raw is None or not isinstance(raw, dict):
        return raw
    from pydantic import ValidationError

    from local_operator.compaction.api import CompactionSettings

    try:
        return CompactionSettings.model_validate(raw)
    except ValidationError as exc:
        print(
            f"\033[1;33mWarning: invalid 'compaction' config, using defaults: {exc}\033[0m",
            file=sys.stderr,
        )
        return CompactionSettings()

#: Sampling knobs copied from an agent record onto ``configure_model`` when
#: the agent sets them. Names match both ``AgentData`` and the committed
#: ``configure_model`` keyword arguments (stream B).
_AGENT_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
)


def resolve_agent(args: argparse.Namespace, agent_registry: Any) -> Any:
    """Resolve the session's agent record, creating it when named.

    Mirrors the legacy ``main()`` behavior: ``--agent-id`` (exec) selects by
    id and fails loudly on a miss; ``--agent``/``--agent-name`` selects by
    name and CREATES the agent when it does not exist yet. Returns ``None``
    for the default ephemeral session.

    Lazy-imports ``AgentEditFields`` so module import never pulls in the
    agent registry's heavy dependencies.
    """
    agent_id = getattr(args, "agent_id", None)
    if agent_id:
        try:
            return agent_registry.get_agent(agent_id)
        except KeyError as exc:
            raise ValueError(f"No agent found with ID: {agent_id}") from exc

    name = getattr(args, "agent_name", None)
    if not name:
        return None
    agent = agent_registry.get_agent_by_name(name)
    if agent is not None:
        return agent

    from local_operator.agents import AgentEditFields  # lazy: heavy module

    return agent_registry.create_agent(
        AgentEditFields(
            name=name,
            security_prompt=None,
            hosting=None,
            model=None,
            description=None,
            last_message=None,
            temperature=None,
            tags=[],
            categories=[],
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


def resolve_hosting_model(
    agent: Any, args: argparse.Namespace, config_manager: Any
) -> tuple[str, str]:
    """Apply the precedence agent > CLI flag > config file.

    Raises ``ValueError`` with the legacy message shapes when either value is
    missing, so the CLI's red-banner handler reports it exactly like before.
    """
    hosting: str | None = getattr(agent, "hosting", None) if agent is not None else None
    hosting = hosting or getattr(args, "hosting", None) or config_manager.get_config_value(
        "hosting"
    )
    model_name: str | None = getattr(agent, "model", None) if agent is not None else None
    model_name = model_name or getattr(args, "model", None) or config_manager.get_config_value(
        "model_name"
    )
    if not hosting:
        raise ValueError("Hosting platform is not configured.")
    if not model_name:
        raise ValueError("Model name is not configured.")
    return hosting, model_name


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Render transcript entries into the LLM-visible message list.

    Thin alias over the engine's single converter
    (:func:`local_operator.session.session._default_convert_to_llm`). Two
    renderings of the same entry type is exactly what let the snapcompact
    path diverge — the host converter replayed the archive's full text while
    dropping the frames, so a compaction pass reduced nothing. One renderer,
    imported, keeps the frame replay and the entry-id passthrough in the
    request path.
    """
    from local_operator.session.session import _default_convert_to_llm

    return _default_convert_to_llm(list(messages))

def _make_request_approval(yolo: bool) -> Callable[[str, str], Awaitable[bool]]:
    """Build the tool-approval gate.

    ``--yolo`` auto-approves every tier (read/write/exec). Otherwise approval
    is an interactive y/N prompt — which can only happen on a tty; headless
    runs deny, so a background job never hangs waiting for input it will
    never get. A non-tty denial is NEVER silent (CL-04): the user must see
    why the tool was rejected and how to change it (``--yolo``).
    """
    if yolo:

        async def auto_approve(tool_name: str, description: str) -> bool:
            return True

        return auto_approve

    async def prompt_approval(tool_name: str, description: str) -> bool:
        if not sys.stdin.isatty():
            print(
                f"approval required but no tty; run with --yolo to auto-approve "
                f"(tool '{tool_name}')",
                file=sys.stderr,
            )
            return False
        try:
            answer = await asyncio.to_thread(
                input, f"Allow tool '{tool_name}' ({description})? [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")

    return prompt_approval


def _latest_user_query(transcript: Any) -> str:
    """Extract the skill-selection query from the transcript.

    Per-turn selection embeds the last user message plus the latest
    compaction summary (docs/REWRITE.md section C). Reads the committed
    ``Transcript.entries()`` shape (``.type``, ``.payload``); any deviation
    degrades to an empty query, which skips selection — skill selection must
    never break a session.
    """
    try:
        entries = transcript.entries()
    except Exception:  # noqa: BLE001 — degradation is the contract
        return ""
    user_text = ""
    summary = ""
    for entry in reversed(entries):
        entry_type = getattr(entry, "type", None)
        payload = getattr(entry, "payload", None) or {}
        if not summary and entry_type == "compaction":
            summary = str(payload.get("summary", "")).strip()
        if not user_text and entry_type == "message" and payload.get("role") == "user":
            content = payload.get("content") or []
            user_text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()
        if user_text and summary:
            break
    return "\n".join(part for part in (user_text, summary) if part)


def _env_details() -> str:
    """Volatile environment facts for the last system block (date rides there
    too, added by ``build_system_blocks``). Kept tiny and byte-stable within
    a run: no timestamps, no process ids."""
    import platform

    return (
        f"Platform: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"Python: {platform.python_version()}\n"
        f"Working directory: {os.getcwd()}"
    )


@dataclass
class _SkillsHooks:
    """Shared mutable state for skills wiring: the built index, the name map,
    and the session-frozen skills block (the first prompt selects; later
    turns reuse it so the system prefix stays byte-stable)."""

    index: Any = None
    by_name: dict[str, Any] = field(default_factory=dict)
    frozen_block: str | None = None


async def _setup_skills(
    credential_manager: Any, config_dir: Path, warnings_out: list[str]
) -> _SkillsHooks:
    """Discovery + index build at session creation (orchestrator duty).

    Steps: roots from ``skills.api.default_skill_roots``, discovery, backend
    selection (API when an embeddings key exists, else the offline local
    embedder), ``SkillIndex`` + eager ``await index.build()``. On ANY failure
    this returns empty hooks and records a warning — skills are optional
    enrichment, never a startup requirement.
    """
    hooks = _SkillsHooks()
    try:
        from local_operator.skills.api import (
            SkillIndex,
            default_backend_from_env,
            default_skill_roots,
            discover_skills,
        )

        skills, discovery_warnings = discover_skills(default_skill_roots(Path.cwd()))
        warnings_out.extend(discovery_warnings)
        hooks.by_name = {skill.name: skill for skill in skills}
        if not skills:
            return hooks

        def get_credential(key: str) -> str | None:
            secret = credential_manager.get_credential(key)
            value = secret.get_secret_value() if secret else ""
            return value or None

        backend = default_backend_from_env(get_credential)
        try:
            hooks.index = SkillIndex(skills, backend, cache_dir=config_dir / "cache")
            await hooks.index.build()
            warnings_out.extend(hooks.index.warnings)
        except Exception as exc:  # noqa: BLE001 — degradation is the contract
            # An unreachable or 500-ing embeddings endpoint degrades SEMANTIC
            # SELECTION only. by_name — the map behind every skill:// read —
            # has no embedding dependency and must stay populated, or a
            # backend outage would turn every skill read into "Unknown
            # skill" for the whole session.
            warnings_out.append(
                f"Skill selection unavailable, continuing with static listing: {exc}"
            )
            hooks.index = None
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        warnings_out.append(f"Skills unavailable, continuing without them: {exc}")
        hooks.index = None
        hooks.by_name = {}
    return hooks


async def _select_skills_block(hooks: _SkillsHooks, query: str) -> str:
    """Session-frozen semantic selection: the FIRST prompt selects, later
    turns reuse the frozen block.

    Per-turn re-selection is incompatible with prompt caching: the skills
    block sits in the system prefix, and any change to it invalidates the
    entire conversation after it on every turn (the cache bench measured
    ~40% stability under per-turn churn). omp resolves this by selecting at
    session start and letting the agent pull deeper context via skill://
    reads (progressive disclosure) — the same contract here: the frozen
    listing names the relevant skills, the read tool fetches their bodies.
    Every failure degrades to an empty block.
    """
    if hooks.frozen_block is not None:
        return hooks.frozen_block
    query = query.strip()
    if not query or hooks.index is None:
        hooks.frozen_block = ""
        return ""
    from local_operator.skills.api import render_block

    picked = await hooks.index.select(query)
    hooks.frozen_block = render_block(picked)
    return hooks.frozen_block


def _make_skill_resolver(hooks: _SkillsHooks) -> Callable[[str], str | None]:
    """Build the ``skill://`` resolver for tools + session.

    Prefers the skills stream's ``make_skill_resolver`` adapter (which
    catches ``ValueError`` and returns the error text as content); falls
    back to an equivalent local adapter until that lands. Non-skill URLs
    return ``None`` so callers can chain resolvers.
    """
    try:
        from local_operator.skills.api import make_skill_resolver  # type: ignore[attr-defined]

        return make_skill_resolver(hooks.by_name)
    except (ImportError, AttributeError):
        pass

    def resolver(url: str) -> str | None:
        if not url.startswith("skill://"):
            return None
        from local_operator.skills.protocol import resolve_skill_url

        try:
            return resolve_skill_url(url, hooks.by_name)
        except ValueError as exc:
            return str(exc)

    return resolver


@dataclass
class _SessionPlan:
    """Everything needed to construct the session, split out so
    ``build_initial_blocks`` can render the startup system prompt without
    instantiating the facade (benchmark hook, orchestrator duty).

    ``auth_store`` rides along (CL-08): callers own its lifetime — folded
    into ``session.dispose`` by :func:`create_session`, closed directly by
    :func:`build_initial_blocks` (which never constructs a session).
    """

    session_kwargs: dict[str, Any]
    system_blocks_provider: Callable[[], Awaitable[list[str]]]
    auth_store: Any = None


def _make_system_blocks_provider(
    tools: list[Any],
    transcript: Any,
    hooks: _SkillsHooks,
) -> Callable[[], Awaitable[list[str]]]:
    """Build the per-turn system-prompt closure.

    Session awaits the result (committed tolerance for awaitable providers),
    so the async skill selection can live inside. Block layout comes from
    ``prompts_api.build_system_blocks``: stable instructions first, then
    tools+skills, then the volatile date/env block last so providers can put
    cache breakpoints on the stable prefix.
    """

    async def provider() -> list[str]:
        from local_operator.prompts_api import build_system_blocks

        query = _latest_user_query(transcript)
        try:
            skills_block = await _select_skills_block(hooks, query)
        except Exception:  # noqa: BLE001 — never break the turn
            skills_block = ""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return build_system_blocks(tools, skills_block, _env_details(), date_str)

    return provider


def _transcript_dir_and_agent_id(
    agent: Any, args: argparse.Namespace, agent_registry: Any
) -> tuple[Path, str]:
    """Pick where this session's JSONL transcript lives (CL-02).

    Legacy ``--train`` semantics:

    - named agent + ``--train`` -> the agent's own directory, so history is
      replayed at startup and appended after each turn;
    - named agent WITHOUT ``--train`` -> an ephemeral per-session directory:
      history is neither replayed from nor appended to the agent dir;
    - no agent but ``--train`` -> the registry's autosave agent (legacy
      ``create_autosave_agent`` semantics);
    - otherwise an ephemeral per-session directory under ``sessions/``: the
      default agent must not persist its session.
    """
    config_dir = Path(agent_registry.config_dir)
    train = bool(getattr(args, "train", False))
    if agent is not None:
        agent_id = str(agent.id)
        if train:
            return config_dir / "agents" / agent_id, agent_id
        session_dir = uuid.uuid4().hex[:12]
        return config_dir / "sessions" / session_dir, agent_id
    if train:
        try:
            autosave = agent_registry.create_autosave_agent()
            agent_id = str(autosave.id)
            return config_dir / "agents" / agent_id, agent_id
        except Exception:  # noqa: BLE001 — fall through to ephemeral
            pass
    session_dir = uuid.uuid4().hex[:12]
    return config_dir / "sessions" / session_dir, "main"


async def _prepare(
    args: argparse.Namespace,
    config_manager: Any,
    credential_manager: Any,
    agent_registry: Any,
    *,
    has_ui: bool,
) -> _SessionPlan:
    """Shared wiring core used by :func:`create_session` and
    :func:`build_initial_blocks`. Returns the Session kwargs plus the blocks
    provider; raises ``ValueError`` when hosting/model config is missing."""
    agent = resolve_agent(args, agent_registry)
    hosting, model_name = resolve_hosting_model(agent, args, config_manager)
    yolo = bool(getattr(args, "yolo", False))

    transcript_dir, agent_id = _transcript_dir_and_agent_id(agent, args, agent_registry)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # --- model + stream fn (stream B contracts) ---------------------------
    from local_operator.env import get_env_config
    from local_operator.model.configure import configure_model, create_stream_fn

    chat_kwargs: dict[str, Any] = {}
    if agent is not None:
        for field_name in _AGENT_SAMPLING_FIELDS:
            value = getattr(agent, field_name, None)
            if value is not None:
                chat_kwargs[field_name] = value
    model_configuration = configure_model(
        hosting=hosting,
        model_name=model_name,
        credential_manager=credential_manager,
        env_config=get_env_config(),
        **chat_kwargs,
    )
    spec = model_configuration.spec

    from local_operator.providers.auth_store import AuthStore

    auth_store = AuthStore(credential_manager=credential_manager)
    stream_fn = create_stream_fn(auth_store, settings=config_manager.get_config().values)

    # --- tools + skills (streams A and C) ---------------------------------
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    config_dir = Path(agent_registry.config_dir)
    skill_warnings: list[str] = []
    hooks = await _setup_skills(credential_manager, config_dir, skill_warnings)
    for warning in skill_warnings:
        print(f"\033[1;33mWarning: {warning}\033[0m", file=sys.stderr)

    request_approval = _make_request_approval(yolo)
    tool_context = ToolContext(
        cwd=os.getcwd(),
        session_id=transcript_dir.name,
        agent_id=agent_id,
        has_ui=has_ui,
        request_approval=request_approval,
    )
    tools = create_tools(tool_context)

    from local_operator.session.transcript import Transcript

    transcript = Transcript(transcript_dir)
    system_blocks_provider = _make_system_blocks_provider(tools, transcript, hooks)

    session_kwargs: dict[str, Any] = dict(
        model=spec,
        stream_fn=stream_fn,
        tools=tools,
        transcript=transcript,
        agent_id=agent_id,
        system_blocks_provider=system_blocks_provider,
        convert_to_llm=default_convert_to_llm,
        compaction_settings=coerce_compaction_settings(
            config_manager.get_config_value("compaction", None)
        ),
        yolo=yolo,
        has_ui=has_ui,
        cwd=os.getcwd(),
        skill_resolver=_make_skill_resolver(hooks),
        request_approval=request_approval,
    )
    return _SessionPlan(
        session_kwargs=session_kwargs,
        system_blocks_provider=system_blocks_provider,
        auth_store=auth_store,
    )


async def wire_mcp_into_session(
    session: Any, builtin_tools: list[Any], cwd: str
) -> Any | None:
    """Load MCP tools and merge them into a constructed session (MCP-20).

    Steps (orchestrator duty, all lazy-imported):

    1. ``discover_and_load_mcp_tools(cwd)`` — startup-gated discovery;
       returns ``(manager, tools, errors)``. Tools include deferred ones
       that await their connection inside ``execute`` (omp semantics).
    2. Merge into the live inventory via ``session.refresh_tools`` — the
       committed hook: full merged set (builtins + MCP), effective from the
       next model call, even mid-turn.
    3. ``manager.set_on_tools_changed`` re-merges on server
       connect/disconnect/list-changed so the inventory tracks MCP state.

    ANY failure degrades to zero MCP tools with a warning — MCP is
    enrichment, never a startup requirement. Returns the manager (caller
    owns its disposal via :func:`attach_mcp_dispose`) or ``None``.
    """
    try:
        from local_operator.mcp import discover_and_load_mcp_tools
    except ImportError:
        print(
            "\033[1;33mWarning: MCP support unavailable, continuing without MCP tools\033[0m",
            file=sys.stderr,
        )
        return None

    try:
        manager, mcp_tools, errors = await discover_and_load_mcp_tools(cwd)
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        print(
            f"\033[1;33mWarning: MCP discovery failed, continuing without MCP tools: "
            f"{exc}\033[0m",
            file=sys.stderr,
        )
        return None

    for entry in errors:
        print(
            f"\033[1;33mWarning: MCP server {entry.get('path', '?')}: "
            f"{entry.get('error', 'unknown error')}\033[0m",
            file=sys.stderr,
        )

    merged = list(builtin_tools) + list(mcp_tools)
    if mcp_tools:
        session.refresh_tools(merged)

    def on_tools_changed(new_mcp_tools: list[Any]) -> None:
        # The manager's callback type tolerates sync handlers; refresh_tools
        # is the atomic swap point (loop re-reads the inventory per call).
        session.refresh_tools(list(builtin_tools) + list(new_mcp_tools))

    manager.set_on_tools_changed(on_tools_changed)
    return manager


def attach_mcp_dispose(session: Any, manager: Any) -> None:
    """Fold ``manager.disconnect_all()`` into the session's dispose path.

    The CLI/TUI/exec all call ``session.dispose()`` exactly once; wrapping
    it here means MCP servers are torn down everywhere without teaching
    every caller about the manager. The wrapper is an instance attribute
    shadowing the method; the manager is also exposed as ``mcp_manager``
    for diagnostics.
    """
    original_dispose = session.dispose

    async def dispose_with_mcp() -> None:
        try:
            await original_dispose()
        finally:
            try:
                await manager.disconnect_all()
            except Exception:  # noqa: BLE001 — teardown must not mask dispose
                pass

    session.dispose = dispose_with_mcp
    session.mcp_manager = manager


def attach_auth_dispose(session: Any, auth_store: Any) -> None:
    """Fold ``auth_store.close()`` into the session's dispose path (CL-08).

    The ``AuthStore`` opens a SQLite connection per session; every front end
    calls ``session.dispose()`` exactly once, so wrapping here guarantees the
    connection (and its file lock) is released everywhere without teaching
    each caller. Composes with :func:`attach_mcp_dispose` — the outermost
    wrapper runs first, so MCP teardown and the store close both happen.
    """
    if auth_store is None:
        return
    original_dispose = session.dispose

    async def dispose_with_auth() -> None:
        try:
            await original_dispose()
        finally:
            try:
                auth_store.close()
            except Exception:  # noqa: BLE001 — teardown must not mask dispose
                pass

    session.dispose = dispose_with_auth


async def create_session(
    args: argparse.Namespace,
    config_manager: Any,
    credential_manager: Any,
    agent_registry: Any,
    *,
    has_ui: bool = False,
) -> "SessionProtocol":
    """Build a fully wired harness session from parsed CLI args.

    This is THE factory shared by ``cli.py`` (interactive TUI / headless
    REPL), ``exec_mode.run_exec`` (foreground exec) and ``exec_worker``
    (background exec). All engine modules are imported lazily inside; the
    caller only needs the three legacy managers plus an argparse namespace
    carrying ``hosting``, ``model``, ``agent_name``/``agent_id``, ``yolo``
    and ``train``.

    Raises ``ValueError`` (caught by the CLI's red-banner handler) when the
    hosting/model configuration is missing.
    """
    from local_operator.session.session import Session
    plan = await _prepare(
        args, config_manager, credential_manager, agent_registry, has_ui=has_ui
    )
    session = Session(**plan.session_kwargs)

    # Auth seam (CL-08): the AuthStore's SQLite connection is owned by this
    # session; fold its close into dispose so every front end releases the
    # file lock on the single ``session.dispose()`` call.
    attach_auth_dispose(session, plan.auth_store)

    # MCP seam (MCP-20): merge discovered MCP tools in, subscribe to live
    # changes, and fold server teardown into session.dispose. Degrades to
    # zero MCP tools on any failure.
    mcp_manager = await wire_mcp_into_session(
        session, list(plan.session_kwargs["tools"]), os.getcwd()
    )
    if mcp_manager is not None:
        attach_mcp_dispose(session, mcp_manager)
    return session


async def build_initial_blocks(
    args: argparse.Namespace,
    config_manager: Any,
    credential_manager: Any,
    agent_registry: Any,
) -> list[str]:
    """Render the session's initial system blocks WITHOUT running a turn.

    Benchmark hook (orchestrator duty): lets
    ``scripts/bench_context_budget.py`` measure the startup prompt size
    (instructions + tools inventory + skills + env) against the <=30k start
    budget without instantiating the session facade.
    """
    plan = await _prepare(args, config_manager, credential_manager, agent_registry, has_ui=False)
    # No session facade is built on this path, so the store's lifetime ends
    # here: close it directly (CL-08) to release the SQLite lock.
    try:
        return await plan.system_blocks_provider()
    finally:
        if plan.auth_store is not None:
            try:
                plan.auth_store.close()
            except Exception:  # noqa: BLE001
                pass
