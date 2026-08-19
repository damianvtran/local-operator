"""Session composition root for the CLI-facing entry points.

Why this module exists: ``cli.py`` (interactive + exec), ``exec_worker.py``
(detached background runs) and later the server facade all need the SAME
wiring from parsed args plus the three legacy managers to a harness
:class:`~local_operator.session.protocol.SessionProtocol`. Centralizing this
wiring keeps precedence rules, transcript-directory policy, lazy knowledge
integration, and the lazy-import discipline in exactly one place.

Constraints honored here (docs/REWRITE.md):

- No module-level imports of providers / session internals / semantic indexes /
  TUI. Every engine import happens inside functions, so importing this module
  is cheap and stays valid while parallel rewrite streams are mid-flight.
- Hosting/model resolution precedence: **agent > CLI flag > config file**
  (the legacy bootstrap order, minus the server-only request overrides).
- User skills and packaged guides are wired end-to-end: discovery + index build
  at session creation, first-task semantic selection, and chained
  ``skill://``/``guide://`` resolution. Any knowledge failure degrades to an
  empty listing with a warning — never a crashed startup.

``create_session`` is async: the TUI's committed factory contract is
``Callable[[], Awaitable[SessionProtocol]]`` and the eager skill-index build
needs an await. Headless callers wrap it in ``asyncio.run``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from local_operator.ansi import sanitize_prompt_line
from local_operator.harness.types import AgentMessage, Message

# Pure path policy, no engine — see local_operator/resume.py for why it is
# its own module rather than living here.
from local_operator.resume import resume_dir

if TYPE_CHECKING:
    # Type-only imports: this module's whole discipline is that the heavy
    # engine, registry and provider modules load lazily inside the functions
    # that need them. Annotations are strings under ``from __future__ import
    # annotations``, so naming the real types here costs nothing at runtime.
    from local_operator.agents import AgentData, AgentRegistry
    from local_operator.compaction.api import CompactionSettings
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.harness.types import AgentTool
    from local_operator.mcp.manager import McpManager
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers.auth_store import AuthStore
    from local_operator.session.goal import GoalState
    from local_operator.session.protocol import SessionProtocol
    from local_operator.session.session import Session
    from local_operator.skills.discovery import Skill
    from local_operator.skills.index import SkillIndex
    from local_operator.variables import VariableStore

logger = logging.getLogger("local_operator.session_factory")

#: Hard cap on the operator's custom instructions, in characters (~16k tokens
#: at 4 chars/token). Generous for hand-written standing rules, small enough
#: that a file pasted over by accident cannot silently consume the context
#: window of every request — the content rides the cached prompt prefix.
MAX_USER_INSTRUCTIONS_CHARS = 64_000

#: Floor held for the selected agent's own profile prompt, so a large global
#: file cannot crowd the chosen profile out entirely. A floor and not a fixed
#: slice: whichever source is smaller than its share leaves the remainder to
#: the other, so neither is taxed for room the other never uses.
_AGENT_INSTRUCTIONS_RESERVE = 16_000


#: Modules whose import dominates :func:`create_session`, measured rather than
#: guessed: on this machine ``mcp`` costs 443 ms and ``httpx`` 234 ms to import,
#: and the engine entries below another 195 ms between them. Everything here is
#: imported lazily by the factory or by something it calls, which is what makes
#: session construction a ~700 ms burst of import machinery.
#:
#: Third-party names sit alongside our own deliberately: the cost is theirs, and
#: naming only our modules would warm the cheap half of the problem.
_WARM_IMPORTS: tuple[str, ...] = (
    "mcp",
    "httpx",
    "httpcore",
    "truststore",
    "local_operator.compaction.api",
    "local_operator.mcp.manager",
    "local_operator.model.configure",
    "local_operator.model.discovery",
    "local_operator.providers.auth_store",
    "local_operator.session.session",
    "local_operator.skills.discovery",
)


def warm_session_imports() -> None:
    """Pay :func:`create_session`'s import cost, off whatever loop is running.

    ``create_session`` is a coroutine, but its body is one long SYNCHRONOUS
    stretch — the awaits are few and none of them yield until the imports are
    done — so a caller with a live event loop is frozen for the whole of it.
    Under the TUI that is a ~700 ms window in which no frame is painted and no
    keypress is handled: the user types the first words of their prompt into a
    screen that does not move, and the characters all appear at once when it
    unfreezes.

    Importing is CPU and file I/O, both of which drop the GIL, so running this
    in a worker thread (``await asyncio.to_thread(warm_session_imports)``)
    turns that one long stall into interleaved sub-frame ones — measured at
    16 ms worst case, against 699 ms for the unwarmed factory. The factory
    itself is unchanged: it still imports what it needs, and finds it cached.

    Never raises. An optional extra that is not installed (``mcp``) or a module
    that fails to import is the factory's problem to report, in the factory's
    own words, at the point where it actually needs it.
    """
    import importlib

    for name in _WARM_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — a warm-up must never be the failure
            logger.debug("prewarm skipped %s", name, exc_info=True)


def coerce_compaction_settings(raw: object) -> CompactionSettings | None:
    """Coerce ``values.compaction`` into a :class:`CompactionSettings` (CL-01).

    ``ConfigManager`` returns the YAML shape verbatim — a plain ``dict`` — but
    the session consumes attribute-style settings. ``None`` and already-typed
    settings pass through; a dict is validated; an invalid dict degrades to
    defaults with a warning (a bad compaction block must never block startup).

    Anything else (``compaction: some-string`` in the YAML) is out of
    contract and reads as "no block": handing the session a junk object would
    only defer the failure to the first compaction check.
    """
    if raw is None:
        return None
    from pydantic import ValidationError

    from local_operator.compaction.api import CompactionSettings

    if isinstance(raw, CompactionSettings):
        return raw
    if not isinstance(raw, dict):
        return None

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


def resolve_agent(args: argparse.Namespace, agent_registry: AgentRegistry) -> AgentData | None:
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
    agent: AgentData | None, args: argparse.Namespace, config_manager: ConfigManager
) -> tuple[str, str]:
    """Apply the precedence agent > CLI flag > config file.

    Raises ``ValueError`` with the legacy message shapes when either value is
    missing, so the CLI's red-banner handler reports it exactly like before.
    """
    hosting: str | None = getattr(agent, "hosting", None) if agent is not None else None
    hosting = (
        hosting or getattr(args, "hosting", None) or config_manager.get_config_value("hosting")
    )
    model_name: str | None = getattr(agent, "model", None) if agent is not None else None
    model_name = (
        model_name or getattr(args, "model", None) or config_manager.get_config_value("model_name")
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


def _fullscreen_app_owns_terminal() -> bool:
    """True when a Textual app currently holds the terminal.

    Reading input from stdin then is not "interactive", it is a DEADLOCK: the
    app has the terminal in raw mode and consumes every keystroke, so a thread
    parked on ``input()`` waits for a line nobody can type and the turn awaiting
    approval never resumes. Probed through Textual's own active-app context var
    (import-guarded, because the TUI is an optional extra and this module sits on
    the headless path too).
    """
    try:
        from textual.app import active_app
    except Exception:  # textual absent: nothing can own the terminal
        return False
    return active_app.get(None) is not None


def _make_request_approval(yolo: bool) -> Callable[[str, str], Awaitable[bool]]:
    """Build the tool-approval gate.

    ``--yolo`` auto-approves every tier (read/write/exec). Otherwise approval
    is an interactive y/N prompt — which can only happen on a tty; headless
    runs deny, so a background job never hangs waiting for input it will
    never get. A non-tty denial is NEVER silent (CL-04): the user must see
    why the tool was rejected and how to change it (``--yolo``).

    A full-screen front end must REPLACE this gate with its own surface
    (``SessionProtocol.set_approval_handler``); the check below is the safety
    net for the window before it does, and for a UI that forgets to. Denying is
    the only safe answer there — the alternative is the hang described in
    :func:`_fullscreen_app_owns_terminal`, which looks to the user like the
    agent froze mid-task.
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
        if _fullscreen_app_owns_terminal():
            # error, not warning: reaching this branch means a front end that owns
            # the terminal did not install an approval handler, which is a wiring
            # BUG, and the user pays for it with a tool that refuses for no
            # visible reason. Named remedies so whoever reads the log can act.
            #
            # Deliberately not stderr, unlike the non-tty branch above: a stray
            # stderr write under a full-screen app paints over the frame and stays
            # there (see tests/unit/tui/test_logger_silence.py), so the CL-04
            # spelling is unavailable here. The TUI routes this file's records to
            # a rotating log, which is where this lands.
            logger.error(
                "approval for %r denied: a full-screen UI owns the terminal and "
                "installed no approval handler — install one via "
                "SessionProtocol.set_approval_handler, or run with --yolo to "
                "auto-approve every tier",
                tool_name,
            )
            return False
        try:
            # Sanitised HERE as well as at the source. This is a second
            # human-facing approval surface, it renders onto a real terminal
            # with no widget between it and the escape codes, and the cost of
            # the belt-and-braces is one function call on a path that is about
            # to block on human input anyway.
            answer = await asyncio.to_thread(
                input,
                "Allow tool '{}' ({})? [y/N] ".format(
                    sanitize_prompt_line(tool_name, limit=120),
                    sanitize_prompt_line(description),
                ),
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
            # A snapcompact entry's summary is reading instructions for the
            # archive frames, not conversation content — as a selection query
            # it is constant boilerplate that would drown the user's actual
            # words. The archive's text_tail is the newest slice of the real
            # transcript, so prefer it (bounded: selection wants a signal, not
            # the whole edge).
            preserve = payload.get("preserve_data") or {}
            snap = preserve.get("snapcompact") if isinstance(preserve, dict) else None
            # Prefer text_tail, then text_head: a small archive stores ALL its
            # text in text_head with an empty tail, and falling straight
            # through to the summary there re-created the boilerplate-noise
            # defect for exactly the sessions with the least other signal.
            edge = ""
            if isinstance(snap, dict):
                for key in ("text_tail", "text_head"):
                    candidate = snap.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        edge = candidate.strip()[-2000:]
                        break
            summary = edge or str(payload.get("summary", "")).strip()
        if not user_text and entry_type == "message" and payload.get("role") == "user":
            content = payload.get("content") or []
            user_text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()
        if user_text and summary:
            break
    return "\n".join(part for part in (user_text, summary) if part)


def _latest_compaction_id(transcript: Any) -> str | None:
    """Entry id of the newest compaction marker, or ``None`` without one.

    This is the freeze key for the knowledge block: selection normally
    freezes after the first query so the prompt-cache prefix stays warm, but
    a compaction rewrites the transcript head anyway — the cache is already
    invalidated — so a NEW id here licenses one re-selection (see
    :func:`_select_knowledge_block`). Same contract as
    :func:`_latest_user_query`: any transcript deviation degrades to
    ``None`` (treated as "no compaction yet"), never breaks the turn.
    """
    try:
        entries = transcript.entries()
    except Exception:  # noqa: BLE001 — degradation is the contract
        return None
    for entry in reversed(entries):
        if getattr(entry, "type", None) == "compaction":
            entry_id = getattr(entry, "id", None)
            return str(entry_id) if entry_id else None
    return None


def _env_details(cwd: str | None = None) -> str:
    """Volatile environment facts for the env block (date rides there too,
    added by ``build_system_blocks``). Kept tiny and byte-stable within a
    run: no timestamps, no process ids. ``cwd`` comes from the session's
    working directory, never the process-global value at call time."""
    import platform

    return (
        f"Platform: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"Python: {platform.python_version()}\n"
        f"Working directory: {cwd if cwd is not None else os.getcwd()}"
    )


def load_user_instructions(agent_prompt: str = "") -> str:
    """Read the operator's standing custom instructions for the system prompt.

    Source of truth is ``<config_dir>/system_prompt.md`` — the same file the
    desktop UI's Settings "Instructions" box and the
    ``/v1/config/system-prompt`` endpoint write, so the three surfaces cannot
    drift into separate notions of "custom instructions".

    ``agent_prompt`` is the selected agent profile's own ``system_prompt.md``.
    It is appended rather than allowed to replace the global file: an agent is
    a specialization ("you review Python"), not a reason to forget the
    operator's machine-wide preferences, and a profile that genuinely must
    override one can say so in its own text.

    Failures degrade instead of breaking startup: an unreadable file is
    skipped, and undecodable bytes are REPLACED rather than dropping the whole
    file, because a stray bad byte in a long instructions file should cost the
    operator one glyph and not every preference they wrote. Either way a bad
    edit never costs a session.

    The result is bounded at :data:`MAX_USER_INSTRUCTIONS_CHARS`. This rides
    the CACHED head block, so it is re-sent as the prefix of every request in
    every session and every subagent: an accidentally huge file (a log pasted
    over the wrong path) would otherwise cost context and money on every call,
    and on a small-context model would fail the session at startup with
    nothing pointing at the cause. Truncation is explicit — the marker tells
    the model its instructions were cut rather than letting it act on half a
    rule — and a warning names the source and the limit.

    ``utf-8-sig`` strips a BOM that a Windows editor writes; without it the
    ``\ufeff`` survives into the prompt ahead of the first rule.
    """
    # Imported as an alias: ``config_dir`` is a local parameter name in two
    # other functions here, and a module-level import of the same spelling
    # would read like one of them.
    from local_operator.paths import config_dir as app_config_dir

    parts: list[str] = []
    # ``is_file()`` follows symlinks deliberately: pointing the file at a
    # dotfiles checkout is a normal way to version instructions.
    path = app_config_dir() / "system_prompt.md"
    try:
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        pass

    # Each source is bounded on its OWN budget before joining. Capping only
    # the joined string let a full-size global file consume the whole budget
    # and silently discard the selected agent's profile prompt entirely,
    # inverting the documented layering: the machine-wide file would discard
    # the profile the operator explicitly chose.
    #
    # The split is a FLOOR each way, never a flat tax. Subtracting the
    # reserve unconditionally cut a 64k global file to 48k even with no agent
    # selected, handing the 16k to nobody; capping the profile at the reserve
    # unconditionally did the mirror image to a large profile when the global
    # file was small. So each source may spend whatever the other leaves,
    # down to its own guaranteed share.
    global_raw = "\n\n".join(part.strip() for part in parts if part.strip())
    agent_raw = agent_prompt.strip()
    # The "\n\n" join is only emitted when BOTH sources survive, so the two
    # characters are only withheld then. Keyed off the agent text alone, a
    # profile that fits the documented cap exactly was truncated by two
    # characters while a global file of the same size passed whole.
    separator = 2 if (agent_raw and global_raw) else 0
    agent_text = _bound_instructions(
        agent_raw,
        "the selected agent's profile",
        max(
            _AGENT_INSTRUCTIONS_RESERVE,
            MAX_USER_INSTRUCTIONS_CHARS - len(global_raw) - separator,
        ),
    )
    global_text = _bound_instructions(
        global_raw,
        str(path),
        MAX_USER_INSTRUCTIONS_CHARS - len(agent_text) - separator,
    )
    return "\n\n".join(part for part in (global_text, agent_text) if part)


def _bound_instructions(text: str, source: str, limit: int) -> str:
    """Cap one instruction source so it cannot silently eat the context window.

    ``source`` names the origin in the warning: passing the global path for
    text that came from an agent profile would send the operator looking for
    a file that may not even exist. The marker is counted INSIDE ``limit``, so
    the return value never exceeds it — including when ``limit`` is too small
    to hold the marker at all, where the marker is dropped rather than
    appended past the budget.
    """
    if len(text) <= limit:
        return text
    marker = f"\n\n[... custom instructions truncated at {limit} characters ...]"
    if limit < len(marker):
        marker = ""
    logger.warning(
        "custom instructions from %s are %d chars; truncating to %d "
        "(they are re-sent with every request)",
        source,
        len(text),
        limit,
    )
    return text[: max(0, limit - len(marker))].rstrip() + marker


def _build_variable_store(cwd: str, config_manager: ConfigManager) -> VariableStore:
    """Construct the session's VariableStore for the list/read variable
    tools. Config ``variables`` ride above the project file and environment;
    no values are ever written into the system prompt (that is the whole
    point — the model lists names and reads single values on demand)."""
    from local_operator.variables import VariableStore

    config_values: dict[str, str] | None = None
    try:
        raw = config_manager.get_config_value("variables", None)
        if isinstance(raw, dict):
            config_values = {str(k): str(v) for k, v in raw.items() if v is not None}
    except Exception:  # noqa: BLE001 — a config read failure must not block tools
        config_values = None
    return VariableStore(cwd=cwd, config_values=config_values)


@dataclass
class _KnowledgeHooks:
    """Session-owned semantic knowledge and progressive-disclosure resolvers.

    User skills and packaged guides share one index. Registered agent metadata
    gets a separate local-only index: it can select the generic agents guide,
    but names and descriptions never enter the prompt or a remote embedding
    request. The selected block freezes after the first task for cache stability.
    """

    index: SkillIndex | None = None
    agent_hint_index: SkillIndex | None = None
    skills_by_name: dict[str, Skill] = field(default_factory=dict)
    guides_by_name: dict[str, Skill] = field(default_factory=dict)
    frozen_block: str | None = None
    #: Compaction entry id observed when ``frozen_block`` was computed. A
    #: change here (a new compaction marker) re-opens selection once — the
    #: transcript head is being rewritten anyway, so the prompt cache the
    #: freeze protects is already gone.
    frozen_compaction_id: str | None = None
    mcp_resolver: Callable[[str], str | None] | None = None
    mcp_catalogue: Callable[[], str] | None = None


def _registered_agent_hints(agent_registry: AgentRegistry) -> list[Skill]:
    """Build bounded, local-only routing rows from meaningful agent metadata.

    A registry can grow indefinitely, and descriptions are user content. Each
    row is capped before hashing/embedding and the first 512 deterministic rows
    are used. Empty autosave-style profiles provide no routing signal and are
    skipped. The rows are never rendered by ``render_block``.
    """
    from local_operator.skills.discovery import Skill

    try:
        agents = sorted(
            agent_registry.list_agents(),
            key=lambda agent: (str(agent.name).lower(), str(agent.name), str(agent.id)),
        )[:512]
    except Exception:  # noqa: BLE001 — hints are optional enrichment
        return []

    hints: list[Skill] = []
    agents_dir = Path(agent_registry.config_dir) / "agents"
    for agent in agents:
        semantic_parts = [
            sanitize_prompt_line(str(agent.description or "")),
            " ".join(sanitize_prompt_line(str(tag)) for tag in (agent.tags or [])),
            " ".join(sanitize_prompt_line(str(category)) for category in (agent.categories or [])),
        ]
        semantic = " ".join(part for part in semantic_parts if part).strip()
        if not semantic:
            continue
        agent_dir = agents_dir / str(agent.id)
        hints.append(
            Skill(
                name=f"registered-agent-{agent.id}",
                description=f"{sanitize_prompt_line(str(agent.name))}: {semantic}"[:512],
                file_path=agent_dir / "agent.yml",
                base_dir=agent_dir,
                source=str(agents_dir),
                resource_type="agent_hint",
            )
        )
    return hints


async def _setup_knowledge(
    credential_manager: CredentialManager,
    config_dir: Path,
    agent_registry: AgentRegistry,
    warnings_out: list[str],
) -> _KnowledgeHooks:
    """Discover and index user skills, packaged guides, and private agent hints.

    Guide bodies are release resources and therefore exist in every install;
    only their short descriptions join the ordinary skill descriptions sent to
    the configured semantic backend. Agent hints always use ``LocalEmbedder``.
    Any layer can fail independently without making session startup fail.
    """
    hooks = _KnowledgeHooks()
    try:
        from local_operator.guides import discover_guides
        from local_operator.skills.api import (
            SkillIndex,
            default_backend_from_env,
            default_skill_roots,
            discover_skills,
        )
        from local_operator.skills.embeddings import LocalEmbedder

        skills, discovery_warnings = discover_skills(default_skill_roots(Path.cwd()))
        warnings_out.extend(discovery_warnings)
        guides = discover_guides()
        hooks.skills_by_name = {skill.name: skill for skill in skills}
        hooks.guides_by_name = {guide.name: guide for guide in guides}
        resources = sorted(
            [*skills, *guides],
            key=lambda item: (
                item.resource_type,
                item.name.lower(),
                item.name,
                str(item.file_path),
            ),
        )

        def get_credential(key: str) -> str | None:
            secret = credential_manager.get_credential(key)
            value = secret.get_secret_value() if secret else ""
            return value or None

        if resources:
            backend = default_backend_from_env(get_credential)
            try:
                hooks.index = SkillIndex(resources, backend, cache_dir=config_dir / "cache")
                await hooks.index.build()
                warnings_out.extend(hooks.index.warnings)
            except Exception as exc:  # noqa: BLE001 — direct reads still work
                hooks.index = None
                if not isinstance(backend, LocalEmbedder):
                    warnings_out.append(
                        f"Knowledge embedding backend failed; using local routing: {exc}"
                    )
                    try:
                        hooks.index = SkillIndex(
                            resources,
                            LocalEmbedder(),
                            cache_dir=config_dir / "cache",
                        )
                        await hooks.index.build()
                    except Exception as fallback_exc:  # noqa: BLE001
                        warnings_out.append(
                            "Knowledge selection unavailable, continuing without routing: "
                            f"{fallback_exc}"
                        )
                        hooks.index = None
                else:
                    warnings_out.append(
                        f"Knowledge selection unavailable, continuing without routing: {exc}"
                    )

        agent_hints = _registered_agent_hints(agent_registry)
        if agent_hints and "agents" in hooks.guides_by_name:
            try:
                hooks.agent_hint_index = SkillIndex(
                    agent_hints,
                    LocalEmbedder(),
                    cache_dir=config_dir / "cache",
                )
                await hooks.agent_hint_index.build()
            except Exception as exc:  # noqa: BLE001 — generic guide routing remains
                warnings_out.append(f"Registered-agent semantic hints unavailable: {exc}")
                hooks.agent_hint_index = None
    except Exception as exc:  # noqa: BLE001 — knowledge is optional enrichment
        warnings_out.append(f"Knowledge guides unavailable, continuing without them: {exc}")
        hooks = _KnowledgeHooks()
    return hooks


async def _select_knowledge_block(
    hooks: _KnowledgeHooks,
    query: str,
    *,
    compaction_id: str | None = None,
    cwd: str | None = None,
) -> str:
    """Freeze selected knowledge plus the bounded MCP catalogue for the session.

    The freeze exists purely for prompt-cache warmth: once the block is
    rendered, later turns reuse it byte-for-byte rather than re-embedding and
    possibly reordering the tail. The ONE thing that re-opens it is a new
    compaction marker (``compaction_id`` differs from the id recorded at
    freeze time): compaction rewrites the transcript head the block rides
    behind, invalidating that cache anyway, so selection is re-run against
    the latest user query + compaction summary (the query the provider
    extracts) and re-frozen under the new id. ``cwd`` feeds gitignore-style
    ``globs`` matching in the index (see
    :meth:`local_operator.skills.index.SkillIndex.select`).
    """
    if hooks.frozen_block is not None and hooks.frozen_compaction_id == compaction_id:
        return hooks.frozen_block

    picked: list[Skill] = []
    query = query.strip()
    if query:
        # cwd rides as a keyword ONLY when set: test doubles and alternate
        # index shapes in the wild implement ``select(query, ...)`` with the
        # historical signature, and there is no globs matching to do without
        # a cwd anyway.
        select_kwargs: dict[str, Any] = {"cwd": Path(cwd)} if cwd else {}
        picked = await hooks.index.select(query, **select_kwargs) if hooks.index is not None else []
        if hooks.agent_hint_index is not None:
            matching_agents = await hooks.agent_hint_index.select(query, k=1)
            agents_guide = hooks.guides_by_name.get("agents")
            if matching_agents and agents_guide is not None and agents_guide not in picked:
                picked.append(agents_guide)
                picked.sort(
                    key=lambda item: (
                        item.resource_type,
                        item.name.lower(),
                        item.name,
                        str(item.file_path),
                    )
                )

    from local_operator.skills.api import render_block

    sections = [section for section in [render_block(picked)] if section]
    if hooks.mcp_catalogue is not None:
        catalogue = hooks.mcp_catalogue()
        if catalogue:
            sections.append(catalogue)
    hooks.frozen_block = "\n\n".join(sections)
    hooks.frozen_compaction_id = compaction_id
    return hooks.frozen_block


def _make_knowledge_resolver(hooks: _KnowledgeHooks) -> Callable[[str], str | None]:
    """Chain lazy knowledge protocols without mixing their namespaces."""
    from local_operator.guides import make_guide_resolver
    from local_operator.skills.api import make_skill_resolver

    guide_resolver = make_guide_resolver(hooks.guides_by_name)
    skill_resolver = make_skill_resolver(hooks.skills_by_name)

    def resolver(url: str) -> str | None:
        guide_result = guide_resolver(url)
        if guide_result is not None:
            return guide_result
        skill_result = skill_resolver(url)
        if skill_result is not None:
            return skill_result
        if hooks.mcp_resolver is not None:
            return hooks.mcp_resolver(url)
        return None

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
    knowledge_hooks: _KnowledgeHooks
    auth_store: AuthStore | None = None


def _make_system_blocks_provider(
    tools: list[AgentTool],
    transcript: Any,
    hooks: _KnowledgeHooks,
    cwd: str | None = None,
    goal_state: "GoalState | None" = None,
    user_instructions: str = "",
    repo_guidance: str = "",
) -> Callable[[], Awaitable[list[str]]]:
    """Build the per-turn system-prompt closure.

    Session awaits the result (committed tolerance for awaitable providers),
    so semantic knowledge selection can live inside. Block layout comes from
    ``prompts_api.build_system_blocks``: stable head (instructions, inventory,
    env) then the session-frozen guide/skill block last, so the cache prefix
    stays warm — re-selected only when a new compaction marker invalidates
    that prefix anyway (see :func:`_select_knowledge_block`).

    ``goal_state`` is the SAME holder the session facade exposes through
    ``set_goal``, which is how a ``/goal`` edit reaches the next turn's
    prompt without rebuilding the session.

    ``user_instructions`` is captured once by the caller and closed over
    rather than re-read here: it lands in the byte-stable head block, so
    re-reading the file per turn would let a mid-session edit silently
    invalidate the whole cached prefix. Editing the file takes effect on
    the next session, which is also what makes a session's prompt reproducible.
    """

    async def provider() -> list[str]:
        from local_operator.prompts_api import build_system_blocks

        query = _latest_user_query(transcript)
        try:
            knowledge_block = await _select_knowledge_block(
                hooks,
                query,
                compaction_id=_latest_compaction_id(transcript),
                cwd=cwd,
            )
        except Exception:  # noqa: BLE001 — never break the turn
            knowledge_block = ""
        date_str = datetime.now().strftime("%Y-%m-%d")
        goal = goal_state.text if goal_state is not None else ""
        return build_system_blocks(
            tools,
            knowledge_block,
            _env_details(cwd),
            date_str,
            goal=goal,
            user_instructions=user_instructions,
            repo_guidance=repo_guidance,
        )

    return provider


def _transcript_dir_and_agent_id(
    agent: AgentData | None, args: argparse.Namespace, agent_registry: AgentRegistry
) -> tuple[Path, str]:
    """Pick where this session's JSONL transcript lives (CL-02).

    ``--resume <id>`` wins over every rule below: it names an existing session
    directory, and reusing it is what makes the transcript replay (the same
    mechanism ``--train`` uses for an agent directory).

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
    resume = getattr(args, "resume", None)
    # `is not None`, not truthiness: `--resume ""` is a user error and must be
    # refused, where silently starting a NEW session would look like a resume
    # that lost the history.
    if resume is not None:
        resumed = resume_dir(config_dir, str(resume))
        return resumed, str(agent.id) if agent is not None else "main"
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
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
    *,
    has_ui: bool,
    cwd: str | None = None,
) -> _SessionPlan:
    """Shared wiring core used by :func:`create_session` and
    :func:`build_initial_blocks`. Returns the Session kwargs plus the blocks
    provider; raises ``ValueError`` when hosting/model config is missing.
    ``cwd`` (default: process cwd) is the single working-directory source for
    the tool context, the session and MCP discovery."""
    agent = resolve_agent(args, agent_registry)
    hosting, model_name = resolve_hosting_model(agent, args, config_manager)
    yolo = bool(getattr(args, "yolo", False))

    transcript_dir, agent_id = _transcript_dir_and_agent_id(agent, args, agent_registry)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Bound the ephemeral session store before this run adds to it. Startup
    # is the only moment at which the live directory is unambiguous, which is
    # what makes "never evict the session that is running" enforceable rather
    # than a race. Best-effort by construction (see retention.sweep_sessions):
    # reclaiming disk must never be the reason a session fails to start.
    from local_operator.session.retention import sweep_from_config

    sweep_from_config(config_manager, Path(agent_registry.config_dir), transcript_dir)

    # Stamp session directories that predate the origin marker, so the
    # ``/resume`` picker stops offering delegated runs on the FIRST launch
    # after an upgrade rather than once natural churn has cleared the store.
    # Runs beside the sweep for the same reason it does: startup is when the
    # store is quiet, and both are best-effort by construction. A no-op on
    # every later launch — each directory is answered once and never
    # re-stamped.
    from local_operator.resume import backfill_session_origins

    backfill_session_origins(Path(agent_registry.config_dir))

    # --- model + stream fn (stream B contracts) ---------------------------
    from local_operator.env import get_env_config
    from local_operator.model.configure import configure_model, create_stream_fn

    chat_kwargs: dict[str, Any] = {}
    if agent is not None:
        for field_name in _AGENT_SAMPLING_FIELDS:
            value = getattr(agent, field_name, None)
            if value is not None:
                chat_kwargs[field_name] = value
    # OFF THE EVENT LOOP. `configure_model` is synchronous, and for a model the
    # shipped registry does not fully describe it fetches the provider's live
    # listing over a BLOCKING httpx client (see
    # `model.configure._info_from_discovery`). On the TUI's loop that is a
    # frozen screen and a swallowed keystroke buffer for as long as the
    # provider takes to answer — up to `discovery.DEFAULT_TIMEOUT_S`, 10 s, on
    # a bad network. A thread costs nothing here: every caller is already
    # awaiting this line, and the work is a network wait plus a memoised
    # lookup.
    model_configuration = await asyncio.to_thread(
        functools.partial(
            configure_model,
            hosting=hosting,
            model_name=model_name,
            credential_manager=credential_manager,
            env_config=get_env_config(),
            **chat_kwargs,
        )
    )
    spec = model_configuration.spec

    from local_operator.providers.auth_store import AuthStore

    auth_store = AuthStore(credential_manager=credential_manager)
    stream_fn = create_stream_fn(
        auth_store,
        settings=config_manager.get_config().values,
        session_id=transcript_dir.name,
    )

    # --- tools + lazy knowledge (streams A and C) --------------------------
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    config_dir = Path(agent_registry.config_dir)
    knowledge_warnings: list[str] = []
    hooks = await _setup_knowledge(
        credential_manager, config_dir, agent_registry, knowledge_warnings
    )
    for warning in knowledge_warnings:
        print(f"\033[1;33mWarning: {warning}\033[0m", file=sys.stderr)

    request_approval = _make_request_approval(yolo)
    effective_cwd = cwd if cwd is not None else os.getcwd()
    # The variables surface behind list_variables/read_variable: config
    # overrides ride above the project file and process environment, and
    # values stay out of the system prompt (read on demand, not baked).
    #
    # Built once and handed to BOTH contexts. The factory context below is what
    # `create_tools` inspects to decide which tools exist; the context a tool
    # actually executes against is rebuilt by `Session._build_tool_context` on
    # every turn, so a store installed only here reached the createIf check and
    # nothing else — `list_variables` advertised itself and then read a bare
    # process-env store, in every session.
    variable_store = _build_variable_store(effective_cwd, config_manager)
    tool_context = ToolContext(
        cwd=effective_cwd,
        session_id=transcript_dir.name,
        agent_id=agent_id,
        has_ui=has_ui,
        request_approval=request_approval,
        variables=variable_store,
        # Role profiles and the ``agent`` tool are backed by this registry; a
        # host without one keeps working off the packaged starters.
        agent_registry=agent_registry,
        web_search_settings=config_manager.get_config_value("web_search", None),
    )
    tools = create_tools(tool_context)

    from local_operator.session.goal import GoalState
    from local_operator.session.transcript import Transcript

    transcript = Transcript(transcript_dir)
    # One holder shared by the prompt provider and the session facade, so a
    # ``/goal`` change lands in the next turn without a session rebuild.
    goal_state = GoalState()
    # Read once, at session construction: see the provider's docstring for why
    # this must not be re-read per turn. A profile's own prompt is layered on
    # top of the global file rather than replacing it.
    agent_prompt = ""
    if agent is not None:
        try:
            agent_prompt = agent_registry.get_agent_system_prompt(str(agent.id))
        # ``ValueError`` covers ``UnicodeDecodeError``, which is NOT an
        # ``OSError``: a mis-encoded profile prompt used to raise straight
        # through here and kill session startup. The registry now reads with
        # ``errors="replace"`` so that specific route can no longer raise, but
        # the guard stays for any other decode path a registry might take —
        # an unreadable profile must never cost the operator their session.
        # Logged rather than swallowed in silence. The guard is deliberately
        # broader than its motivating decode error, because a profile prompt is
        # not worth a failed session whatever the registry raises reading it --
        # but a session that quietly drops the agent the operator selected
        # looks like the profile was empty, so the reason has to be findable.
        except (KeyError, OSError, ValueError) as exc:
            logger.warning(
                "could not read the system prompt for agent %s (%s: %s); "
                "continuing without the profile's own instructions",
                agent.id,
                type(exc).__name__,
                exc,
            )
            agent_prompt = ""
    user_instructions = load_user_instructions(agent_prompt)
    # Repo guidance (AGENTS.md/CLAUDE.md ancestors) joins the same read-once
    # contract: the head block must stay byte-stable for the session, so the
    # filesystem is consulted here and never again.
    from local_operator.context_files import load_repo_guidance

    try:
        repo_guidance = load_repo_guidance(effective_cwd)
    except Exception:  # noqa: BLE001 — never block session construction
        repo_guidance = ""

    system_blocks_provider = _make_system_blocks_provider(
        tools,
        transcript,
        hooks,
        cwd=effective_cwd,
        goal_state=goal_state,
        user_instructions=user_instructions,
        repo_guidance=repo_guidance,
    )

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
        cwd=effective_cwd,
        # Session keeps the historical parameter name, but the chained
        # resolver handles both guide:// and skill:// without namespace leaks.
        skill_resolver=_make_knowledge_resolver(hooks),
        request_approval=request_approval,
        goal_state=goal_state,
        variables=variable_store,
        agent_registry=agent_registry,
    )
    return _SessionPlan(
        session_kwargs=session_kwargs,
        system_blocks_provider=system_blocks_provider,
        knowledge_hooks=hooks,
        auth_store=auth_store,
    )


async def wire_mcp_into_session(
    session: Session,
    builtin_tools: list[AgentTool],
    cwd: str,
    knowledge_hooks: _KnowledgeHooks | None = None,
    auth_store: AuthStore | None = None,
    *,
    has_ui: bool = False,
) -> McpManager | None:
    """Discover MCPs but expose their schemas only after explicit reads.

    Startup connects and caches servers exactly as before, but the session
    begins with its non-MCP tools only. A bounded ``<mcps>`` catalogue tells
    the model which servers exist. ``read mcp://<server>`` lists tools without
    loading schemas; ``read mcp://<server>/<tool>`` activates exactly one tool.
    Live list-changed events refresh only schemas the model already selected.
    This keeps an unused MCP server at O(server names), not O(all tool schemas),
    on every provider request.

    ``has_ui`` selects how failures are announced, never whether they are
    recorded. Full-screen clients read ``session.mcp_startup``; headless callers
    get the warning on stderr. Any failure degrades to an empty catalogue, so
    MCP enrichment never becomes a session startup requirement. Returns the
    manager for the caller to dispose, or ``None``.
    """
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY, McpStartupOutcome

    if knowledge_hooks is None:
        knowledge_hooks = _KnowledgeHooks()

    try:
        from local_operator.mcp import discover_and_load_mcp_tools
        from local_operator.mcp.manager import MCP_SDK_MISSING_ERROR
    except ImportError:
        if not has_ui:
            print(
                "\033[1;33mWarning: MCP support unavailable, continuing without MCP tools\033[0m",
                file=sys.stderr,
            )
        # This does NOT catch a missing MCP SDK. Every SDK import in the package
        # is either ``TYPE_CHECKING`` or function-local, so ``local_operator.mcp``
        # imports cleanly with the SDK absent and that case lands in the error
        # loop below instead. What reaches here is our OWN package failing to
        # import — a partial or broken install. An EMPTY outcome is still the
        # right record for it: without the config layer we cannot read the config
        # files, so we do not know whether this machine wanted MCP at all, and
        # "MCP is broken" on a host that never used it is noise.
        session.mcp_startup = McpStartupOutcome()
        return None

    try:
        manager, mcp_tools, errors = await discover_and_load_mcp_tools(cwd, auth_store=auth_store)
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        # Discovery raising IS reportable, unlike the import gap above: reaching
        # this line means the config layer was present and still could not be
        # read, so the user has an MCP setup that is not working.
        if not has_ui:
            print(
                f"\033[1;33mWarning: MCP discovery failed, continuing without MCP tools: "
                f"{exc}\033[0m",
                file=sys.stderr,
            )
        session.mcp_startup = McpStartupOutcome(failures={MCP_DISCOVERY_KEY: str(exc)})
        return None

    # One pass over the error entries: the record keys on the BARE server name
    # (the discovery wrapper reports paths as ``mcp:<server>``) because that is
    # what the user typed in ``.mcp.json`` and what ``/mcp`` lists back. Entries
    # WITHOUT that prefix are the layer failing rather than a server — the
    # wrapper's synthetic hard-failure entry says ``.mcp.json`` — so they take
    # the same key the raising arm above uses. One synthetic key, not three
    # spellings of "not a server".
    failures: dict[str, str] = {}
    for entry in errors:
        path = str(entry.get("path", "?"))
        message = str(entry.get("error", "unknown error"))
        failures[path.partition("mcp:")[2] or MCP_DISCOVERY_KEY] = message

    if failures and set(failures.values()) == {MCP_SDK_MISSING_ERROR}:
        # The SDK is not installed, so the manager failed every configured server
        # with the same install instruction. Reported ONCE, as the setup problem
        # it is: N identical 90-character notices (one toast line plus one
        # transcript error per server, every launch) is noise proportional to
        # server count for a single cause, and it accuses the servers of a fault
        # that is not theirs. Compared by identity against the manager's own
        # constant rather than by substring, so re-wording it cannot silently
        # disable this.
        failures = {MCP_DISCOVERY_KEY: MCP_SDK_MISSING_ERROR}

    if not has_ui:
        for name, message in failures.items():
            subject = "MCP discovery" if name == MCP_DISCOVERY_KEY else f"MCP server {name}"
            print(f"\033[1;33mWarning: {subject}: {message}\033[0m", file=sys.stderr)

    session.mcp_startup = McpStartupOutcome(
        configured=tuple(manager.get_all_server_names()),
        connected=tuple(manager.get_connected_servers()),
        failures=failures,
        tool_count=len(mcp_tools),
    )

    # The NON-MCP base is READ BACK from the session on every refresh, not
    # snapshotted once. Session capability tools live in that inventory even
    # though ``builtin_tools`` predates them, and some of them are merged in
    # AFTER this wiring runs: the TUI installs its ask handler in
    # ``_adopt_session``, long after the factory returned, and a frozen base
    # would silently un-advertise ``ask`` again the first time the model
    # activated any MCP tool. What is subtracted is the set this function last
    # installed itself, so classification never depends on ``get_tool_meta``
    # still answering for a server that has since dropped away. The snapshot
    # below survives only as the fallback for a host that exposes no inventory
    # to read back.
    base_inventory = list(
        getattr(session, "_tools", None) or getattr(session, "tools", None) or builtin_tools
    )
    installed_mcp: set[str] = set()
    enabled_origins: set[tuple[str, str]] = set()

    def selected_tools(source: list[AgentTool]) -> list[AgentTool]:
        selected: list[AgentTool] = []
        for tool in source:
            meta = manager.get_tool_meta(tool.name) or {}
            origin = (str(meta.get("server_name", "")), str(meta.get("mcp_tool_name", "")))
            if origin in enabled_origins:
                selected.append(tool)
        return selected

    def refresh_selected(source: list[AgentTool]) -> None:
        live = list(getattr(session, "_tools", None) or getattr(session, "tools", None) or ())
        base = [tool for tool in live if tool.name not in installed_mcp] or base_inventory
        selected = selected_tools(source)
        installed_mcp.clear()
        installed_mcp.update(tool.name for tool in selected)
        session.refresh_tools(base + selected)

    def activate(server_name: str, raw_tool_name: str) -> None:
        enabled_origins.add((server_name, raw_tool_name))
        refresh_selected(manager.get_tools())

    from local_operator.mcp.resources import make_mcp_resolver, render_mcp_catalogue

    knowledge_hooks.mcp_resolver = make_mcp_resolver(manager, activate)
    knowledge_hooks.mcp_catalogue = lambda: render_mcp_catalogue(manager)

    def on_tools_changed(new_mcp_tools: list[AgentTool]) -> None:
        # Reconnects and tools/list_changed can replace AgentTool objects. Keep
        # the selected origins and swap in only their fresh schemas.
        refresh_selected(new_mcp_tools)

    manager.set_on_tools_changed(on_tools_changed)
    return manager


def attach_mcp_dispose(session: Session, manager: McpManager) -> None:
    """Fold ``manager.disconnect_all()`` into the session's dispose path.

    The CLI/TUI/exec all call ``session.dispose()`` exactly once, so hanging
    MCP teardown off it tears the servers down everywhere without teaching
    each caller about the manager. The manager is also exposed as
    ``mcp_manager`` for diagnostics.
    """
    session.add_dispose_hook(manager.disconnect_all)
    session.mcp_manager = manager
    # Breaker incidents become session incidents: the model learns a server's
    # tools are gone instead of hammering them (MCP-07's observable half).
    manager.on_incident = session._on_mcp_incident


def attach_auth_dispose(session: Session, auth_store: AuthStore | None) -> None:
    """Fold ``auth_store.close()`` into the session's dispose path (CL-08).

    The ``AuthStore`` opens a SQLite connection per session; every front end
    calls ``session.dispose()`` exactly once, so registering here guarantees
    the connection (and its file lock) is released everywhere without
    teaching each caller.
    """
    if auth_store is None:
        return
    session.add_dispose_hook(auth_store.close)


def attach_stream_dispose(session: Session, stream_fn: SessionStreamFn) -> None:
    """Fold the session's shared ``httpx.AsyncClient`` close into dispose.

    ``create_stream_fn`` builds one client per session and hangs its close on
    the returned object; without this seam the pool leaks for the process
    lifetime (one per turn on the server facade).
    """
    session.add_dispose_hook(stream_fn.close)


async def create_session(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
    *,
    has_ui: bool = False,
    cwd: str | None = None,
) -> "SessionProtocol":
    """Build a fully-wired harness session from parsed CLI args.

    This is THE factory shared by ``cli.py`` (interactive TUI / headless
    REPL), ``exec_mode.run_exec`` (foreground exec) and ``exec_worker``
    (background exec). All engine modules are imported lazily inside; the
    caller only needs the three legacy managers plus an argparse namespace
    carrying ``hosting``, ``model``, ``agent_name``/``agent_id``, ``yolo``
    and ``train``.

    ``cwd`` is the session's working directory; ``None`` means the process
    cwd (legacy behaviour). Hosts that must relocate a session (the
    scheduler's per-agent directory) pass it explicitly instead of mutating
    the process-global cwd across awaits — every other session builder in
    the same process would otherwise read the wrong directory.

    Raises ``ValueError`` (caught by the CLI's red-banner handler) when the
    hosting/model configuration is missing.
    """

    from local_operator.session.session import Session

    effective_cwd = cwd if cwd is not None else os.getcwd()
    plan = await _prepare(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=has_ui,
        cwd=effective_cwd,
    )
    session = Session(**plan.session_kwargs)

    # Auth seam (CL-08): the AuthStore's SQLite connection is owned by this
    # session; fold its close into dispose so every front end releases the
    # file lock on the single ``session.dispose()`` call.
    attach_auth_dispose(session, plan.auth_store)
    # Stream seam: release the session's shared httpx connection pool on
    # dispose (one leaked pool per turn on the server facade otherwise).
    attach_stream_dispose(session, plan.session_kwargs["stream_fn"])

    # MCP seam (MCP-20): merge discovered MCP tools in, subscribe to live
    # changes, and fold server teardown into session.dispose. Degrades to
    # zero MCP tools on any failure. ``has_ui`` routes the announcement: a
    # front end with a full-screen terminal reads session.mcp_startup instead
    # of being written over by a stderr warning.
    mcp_manager = await wire_mcp_into_session(
        session,
        list(plan.session_kwargs["tools"]),
        effective_cwd,
        knowledge_hooks=plan.knowledge_hooks,
        auth_store=plan.auth_store,
        has_ui=has_ui,
    )
    if mcp_manager is not None:
        attach_mcp_dispose(session, mcp_manager)
    return session


async def build_initial_blocks(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
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
