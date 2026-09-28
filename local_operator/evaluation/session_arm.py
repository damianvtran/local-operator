"""The benchmark's session engagement: an episode that runs AS a session.

WHAT THIS IS. Design ``docs/design/sdk-engagement.md`` §6: the benchmark stops
hand-building a reply channel and drives the SAME session object the TUI,
``lop exec`` and the mobile runtime drive, with the episode's computer-use
actions exported as a per-session MCP server (``action_server.py``). This
module is the engagement glue: it declares that server in the episode's own
scratch root, hosts the bridge the server forwards calls to, opens the session
through ``local_operator.sdk``, records the one event stream, and (in
``run_session_episode``) drives one whole episode -- launch, reset, session
turn, score, cleanup, close.

WHY THIS MODULE IS NOT IN ``runner/``. The runner core (``evaluation/runner``)
must never import the application: an episode has to be reproducible from its
pinned inputs, and the isolation test asserts that boundary. This module is
deliberately on the OTHER side of that boundary -- it exists precisely to hand
the episode the real session -- so nothing under ``runner/`` may import it,
and it lives beside the evaluation package's other consumer-facing modules.
The old arm imports nothing from here; the two modes are chosen by the run
script and the reply channel's semantics are untouched.

THE TWO HALVES OF THE BRIDGE. ``ActionBridge`` (episode side) owns the token
discipline, the step budget, and the execution seam; the MCP server (child
side) owns nothing but the wire. Calls arrive one per connection, are
serialized here, and carry exactly the shape ``runner/action_tool.py``
projects -- the batch models, the refusals and their hints are imported from
there rather than re-stated, so the MCP channel and the loop-driven channel
cannot drift apart.

STATUS VOCABULARY (the pilot arm's record, not the sealed-bundle format).
``run_session_episode`` returns a :class:`SessionArmOutcome` with one of:
``completed`` (the model declared finish), ``agent_stop`` (the turn ended
without a terminal batch), ``truncated`` (the step budget or the wall bound
ended the run; the state reached is still scored), ``failed_pre_bundle`` (the
environment never came up), ``failed`` (the run died mid-episode). The record
is a directory with ``events.jsonl`` (the session's own event stream, via
``headless_print.printable_event`` -- the same projection ``exec --json``
prints), ``outcome.json``, and ``score.json``. It is deliberately NOT the
sealed evidence-bundle format: converging the pilot's record onto the campaign
format is the next arm's decision, and claiming comparability early would be
the one thing a benchmark must not do.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from local_operator import sdk
from local_operator.evaluation.action_server import (
    SERVER_NAME,
    decode_call,
    encode_response,
    surface_to_json,
)
from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AskUserExchangeParams,
    CleanupParams,
    CloseParams,
    ExecuteParams,
    Handshake,
    InspectRequirementsParams,
    PrepareParams,
    RequirementsResult,
    RescueDescriptor,
    ResetStartParams,
    ResolvedSecret,
    ScoreParams,
)
from local_operator.evaluation.adapters.supervisor import (
    AdapterSupervisor,
    HostVerifier,
    VerifiedAdapterSession,
    discard_rescue,
    persist_rescue,
    run_rescue,
)
from local_operator.evaluation.lifecycle import (
    CleanupAction,
    CleanupPlan,
    aggregate_cleanup,
)
from local_operator.evaluation.protocol import ActionBatch, Observation
from local_operator.evaluation.runner.action_tool import (
    ACTION_TOOL_NAME,
    PendingObservationToken,
    _build_batch,
    _no_pending_refusal,
    _refusal,
)
from local_operator.evaluation.runner.episode import (
    _PROVISIONAL_CLEANUP_ACTION,
    EpisodeConfig,
    EpisodeSpec,
    UndeclaredDisclosedInfra,
    _observation_phase_failure,
    _terminal_kind,
)
from local_operator.evaluation.runner.model import EpisodeTurn
from local_operator.evaluation.runner.provider_client import (
    DEFAULT_KEEP_RECENT_FRAMES,
    DEFAULT_REBUILD_EVERY_FRAMES,
    _ContextBuilder,
)
from local_operator.harness.types import AgentEvent, ImageContent, TextContent
from local_operator.headless_print import printable_event
from local_operator.mcp.config import load_all_mcp_configs
from local_operator.mcp.tool_bridge import create_mcp_tool_name
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec

#: The engagement mode's identifier in the run script, and in every record this
#: module writes. Spelled once so a comparison can never read two spellings of
#: the same arm as two arms.
SESSION_ARM_ID = "session"

#: How the episode's session is asked to behave. ``auto()`` is exec ``--yolo``
#: semantics -- the same posture the repo's own benchmark drives the product
#: with (``docs/BENCHMARKS.md``: "one ``local_operator.cli exec --json --yolo``
#: run per task"), and the honest posture for an unattended episode: a gate
#: that refuses a call "because nobody can be asked" would be measuring the
#: harness's refusal, not the model. The reply-channel arm has no gates at
#: all, so auto() is also its closest parity.
SESSION_APPROVALS = ApprovalPolicy.auto()

#: The first prompt's protocol paragraph. The action tool's own description
#: (shared with the loop-driven tool) carries the per-call contract; this adds
#: only what the episode knows: which tool name the actions reached the session
#: as, and how the episode ends. Everything else the model needs is on the
#: schema.
PROMPT_HEADER = (
    "Act on this computer by calling `{tool_name}`: each call carries ONE batch "
    "of actions for the screen you were just shown and returns the state that "
    "results. Send exactly one call per turn. When the task is complete, call "
    "`{tool_name}` with a single finish action.\n"
)

#: The sentence a terminal batch's tool result carries. Stated as the reason
#: the episode is ending rather than as a refusal, because a ``finish`` batch
#: is a deliberate ending -- the runner's step loop returns on it without ever
#: executing it (``_step_loop``), and the pilot keeps that semantic.
FINISH_ACK = (
    "Episode finished: the finish action was recorded and the run will be "
    "scored on the state reached. Do not call this tool again."
)

#: The step budget's terminal sentence. The runner treats reaching ``max_steps``
#: as a TRUNCATION of a normal, still-scored episode (``_step_loop``); this is
#: the model-facing half of that same decision, and the driver records the
#: truncation itself.
BUDGET_ACK = (
    "Episode step budget reached: the environment will not accept further "
    "actions and the run will be scored on the state reached. Do not call this "
    "tool again."
)


class SessionArmError(RuntimeError):
    """A precondition of the session engagement the caller must fix."""


# ---------------------------------------------------------------------------
# The MCP declaration: one server, in the episode's own scratch root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionServerDeclaration:
    """What was declared: the identity and the exact argv the session dials."""

    server_name: str
    tool_name: str
    config_path: Path
    endpoint: Path
    command: str
    args: tuple[str, ...]


def declare_action_server(
    *,
    config_dir: Path,
    endpoint: Path,
    surface: ActionSurface,
    python_executable: str | None = None,
    cwd: Path | None = None,
) -> ActionServerDeclaration:
    """Write the episode's ``mcp.json`` declaration; return its resolved facts.

    The declaration goes in the SESSION ROOT's own user config file
    (``<config_dir>/mcp.json``), which is the file
    ``mcp/config.load_all_mcp_configs`` reads as the user-scope source. Writing
    it anywhere else -- a shared file, the operator's tree, or the cwd's
    project slot -- would make the episode's tool surface depend on something
    outside ``SessionRoots``; the caller asserts that consequence with
    :func:`assert_declaration_resolved`.

    An EXISTING file is merged, not replaced: a resumed episode or a future
    multi-server episode keeps its other declarations, and only the one name
    this module owns is replaced. The write is atomic (same-directory temp +
    ``os.replace``), the discipline ``mcp/config.py`` itself uses, so a crash
    can never leave a half-written file that the session would read as "no
    servers".
    """

    executable = python_executable or sys.executable
    tool_name = create_mcp_tool_name(SERVER_NAME, ACTION_TOOL_NAME)
    args = [
        "-m",
        "local_operator.evaluation.action_server",
        "--endpoint",
        str(endpoint),
        "--surface",
        surface_to_json(surface),
    ]
    entry: dict[str, Any] = {
        "type": "stdio",
        "command": executable,
        "args": args,
        # The action server must be visible WITHOUT a discovery read: the
        # episode's whole point is that the model acts, and a tool the model
        # cannot see is a tool it does not use (``mcp/config.py``'s own reason
        # for ``preloadTools``). The allowlist pins the surface to the one
        # tool, so a future server-side addition cannot silently widen an
        # episode's tools.
        "preloadTools": True,
        "enabledTools": [ACTION_TOOL_NAME],
    }
    if cwd is not None:
        entry["cwd"] = str(cwd)

    config_path = Path(config_dir) / "mcp.json"
    document: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SessionArmError(
                f"the episode's mcp.json at {config_path} is not readable JSON, so the "
                f"action server cannot be declared without discarding it: {error}"
            ) from error
        if not isinstance(existing, dict):
            raise SessionArmError(f"the episode's mcp.json at {config_path} is not an object")
        document = existing

    servers = document.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_NAME] = entry
    document["mcpServers"] = servers

    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = config_path.with_name(f".{config_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(config_path)

    return ActionServerDeclaration(
        server_name=SERVER_NAME,
        tool_name=tool_name,
        config_path=config_path,
        endpoint=Path(endpoint),
        command=executable,
        args=tuple(args),
    )


def assert_declaration_resolved(
    *,
    cwd: Path,
    config_dir: Path,
    scratch_root: Path,
    server_name: str = SERVER_NAME,
) -> Mapping[str, str]:
    """Prove, by resolved path, that the episode's MCP graph is inside scratch.

    THE RECORDED MUST (PR 1's QA round-1 checklist, design §5): MCP discovery
    is cwd-scoped, so an episode held at the operator's home would enumerate
    and dial the operator's real servers. This is the assertion that makes the
    consequence checkable at run time and in tests:

    * the ``<cwd>/.local-operator/mcp.json`` project slot resolves inside the
      scratch root (the comment above states why cwd may never be elsewhere);
    * the user slot resolves to the episode's own config file;
    * every SOURCE the discovery actually read is under the scratch root -- the
      check that catches an import (``~/.claude.json``, ``~/.cursor/mcp.json``,
      ``~/.codex/config.toml``) resolving somewhere real through a redirected
      or unredirected home.

    Returns ``(name -> source path)`` for the record; raises otherwise.
    """

    scratch = Path(scratch_root).resolve()
    resolved_cwd = Path(cwd).resolve()
    resolved_config = Path(config_dir).resolve()
    if not (resolved_cwd == scratch or resolved_cwd.is_relative_to(scratch)):
        raise SessionArmError(
            f"the episode's cwd {resolved_cwd} is not inside the scratch root {scratch}; "
            "MCP discovery is cwd-scoped and would read the operator's own tree"
        )
    if not (resolved_config == scratch or resolved_config.is_relative_to(scratch)):
        raise SessionArmError(
            f"the episode's config dir {resolved_config} is not inside the scratch root "
            f"{scratch}; the action server would be declared outside the episode"
        )

    configs, sources = load_all_mcp_configs(resolved_cwd)
    outside = {
        name: source
        for name, source in sources.items()
        if not Path(source).resolve().is_relative_to(scratch)
    }
    if outside:
        raise SessionArmError(
            f"this episode's MCP configuration resolves outside its scratch root: {outside}"
        )
    if server_name not in configs:
        raise SessionArmError(
            f"the action server {server_name!r} was not discovered from {resolved_cwd}; "
            "the declaration and the session's cwd disagree"
        )
    return sources


# ---------------------------------------------------------------------------
# Rendering: one definition of what the model is shown
# ---------------------------------------------------------------------------


class ObservationRenderer:
    """An observation as model content, rendered by the reply arm's builder.

    The reply-channel arm renders observations through
    ``provider_client._ContextBuilder`` (text lines + frame image blocks read
    through ``verify_artifact``). The session arm must not grow a SECOND
    definition of "what the model is shown" -- the two arms would drift on
    exactly the facts a comparison depends on -- so the builder itself is
    reused here: one turn appended per observation, and the newest message IS
    the content for the prompt (observation zero) or the tool result (every
    later observation).
    """

    def __init__(self, artifact_root: Path) -> None:
        self._builder = _ContextBuilder(
            artifact_root=artifact_root,
            keep_recent_frames=DEFAULT_KEEP_RECENT_FRAMES,
            rebuild_every_frames=DEFAULT_REBUILD_EVERY_FRAMES,
        )
        self._turns: list[EpisodeTurn] = []

    def render(self, observation: Observation) -> list[Any]:
        """The observation's content blocks, in the builder's own rendering."""

        self._turns.append(EpisodeTurn(observation=observation))
        self._builder.append_new_turns(self._turns)
        message = self._builder.messages[-1]
        return list(message.content)


def split_prompt_content(blocks: Sequence[Any]) -> tuple[str, list[ImageContent]]:
    """Split rendered content into (text, images) for ``Session.prompt``."""

    texts: list[str] = []
    images: list[ImageContent] = []
    for block in blocks:
        if isinstance(block, ImageContent):
            images.append(block)
        else:
            texts.append(block.text)
    return "\n".join(texts), images


# ---------------------------------------------------------------------------
# The bridge: the episode half of the action wire
# ---------------------------------------------------------------------------

#: Called for every executed batch; the driver records it. Receives the batch
#: and the adapter's result.
RecordBatch = Callable[[str, Mapping[str, Any]], None]

#: The execution seam. The driver fills it with the adapter call plus its
#: read-back recovery; tests fill it with anything that looks like
#: ``ExecuteResult``.
ExecuteBatch = Callable[[ActionBatch], Awaitable[Any]]


@dataclass
class ActionBridge:
    """One episode's action bridge: token, budget, execution, refusals.

    The state machine is ``runner/action_tool.py``'s, per the design's rule
    that the session arm keeps the SAME action contract: one batch per turn,
    validate-before-consume, frames rendered by the one renderer, refusals
    carrying the shared vocabulary's class and hint. What is different is only
    WHERE it lives -- behind an MCP tool call instead of a loop-injected tool
    -- because a session cannot have tools injected into its LoopContext by a
    consumer (that would be exactly the harness coupling the design forbids).
    """

    endpoint: Path
    surface: ActionSurface
    render: Callable[[Observation], list[Any]]
    execute: ExecuteBatch
    max_steps: int
    record: RecordBatch | None = None
    ask: Callable[[ActionBatch], Awaitable[str | None]] | None = None

    #: Set when the bridge has decided the episode should end (``finish``, the
    #: step budget). The driver reads it after the turn: a reason here is the
    #: episode's own terminal, and the driver will not re-prompt.
    end_requested: str | None = None

    def __post_init__(self) -> None:
        # The token is armed with observation zero, which only exists once the
        # environment has reset -- so the bridge is constructed ARMED LATER
        # (``arm``) rather than with a placeholder. Serving before arming is a
        # driver bug and is answered as a plain refusal, not an exception into
        # the socket handler.
        self._token: PendingObservationToken | None = None
        self._steps = 0
        self._lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None
        self._closed = False

    def arm(self, observation: Observation) -> None:
        """Arm the token with the episode's first observation."""

        if self._token is not None:
            raise SessionArmError("the action bridge was already armed with an observation")
        self._token = PendingObservationToken(observation)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def terminal(self) -> bool:
        return self._token is not None and self._token.terminal

    def fold(self, event: AgentEvent) -> None:
        """Fold session events; the turn boundary re-arms the token.

        Folding is the DRIVER's job (as in the loop-driven design): the
        engine does not know the token exists, so the arming happens on
        ``TurnEndEvent`` and the next call reads the result.
        """

        if self._token is not None:
            self._token.fold(event)

    async def start(self) -> None:
        self.endpoint.parent.mkdir(parents=True, exist_ok=True)
        # The wire is a UNIX socket inside the episode scratch: nothing off the
        # machine can reach it, and nothing outside the scratch can guess it.
        # Path length is the one platform trap (sockaddr_un is ~104 bytes on
        # macOS), so an over-long path fails loudly here rather than as a
        # confusing connect error inside the child.
        if len(str(self.endpoint).encode("utf-8")) >= 100:
            raise SessionArmError(
                f"action bridge endpoint {self.endpoint} is too long for a UNIX socket; "
                "put the episode scratch on a shorter path"
            )
        self._server = await asyncio.start_unix_server(self._handle, path=str(self.endpoint))

    async def stop(self) -> None:
        self._closed = True
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        try:
            self.endpoint.unlink()
        except OSError:
            pass

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request = decode_call(line)
            reply = await self.call(request.get("arguments") or {})
            writer.write(encode_response(**reply))
            await writer.drain()
        except Exception as error:  # noqa: BLE001 - every frame gets an answer
            # The server validates the reply before the model sees it; a broken
            # frame is answered with an error result rather than killing the
            # bridge -- the next call must still find a live endpoint. Broad on
            # purpose: the caller is a child process waiting on a line, and a
            # driver-side bug must surface as a tool result it can act on, not
            # as a hung socket.
            try:
                writer.write(
                    encode_response(
                        [
                            TextContent(
                                text=(
                                    "action bridge error: the episode driver could not "
                                    f"answer this call ({type(error).__name__})"
                                )
                            )
                        ],
                        is_error=True,
                    )
                )
                await writer.drain()
            except (OSError, ValueError):
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def call(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """One model call, decided exactly as the loop-driven tool decides it."""

        async with self._lock:
            call_id = f"step-{self._steps}-{uuid.uuid4().hex[:6]}"
            token = self._token
            pending = token.pending if token is not None else None
            if token is None or pending is None or token.terminal:
                refusal = _no_pending_refusal(call_id)
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }
            from pydantic import ValidationError

            from local_operator.evaluation.runner.provider_client import (
                classify_admission_error,
                classify_validation_error,
                validation_diagnostic,
            )

            try:
                batch = _build_batch(arguments, pending)
            except ValidationError as error:
                refusal = _refusal(
                    call_id,
                    validation_diagnostic(error),
                    class_key=classify_validation_error(error),
                    observation=pending,
                    surface=self.surface,
                )
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }
            try:
                batch.validate_for(pending)
                self.surface.validate_batch(batch)
            except ValueError as error:
                refusal = _refusal(
                    call_id,
                    f"action batch does not match this observation: {error}",
                    class_key=classify_admission_error(error),
                    observation=pending,
                    surface=self.surface,
                )
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }

            if self._steps >= self.max_steps:
                # The runner's own ordering: the budget is checked BEFORE the
                # model's next decision, so batch N+1 never runs. The episode
                # is still scored on the state it reached.
                token.mark_terminal()
                self.end_requested = self.end_requested or "max-steps"
                if self.record is not None:
                    self.record("budget", {"steps": self._steps})
                return {
                    "content": [TextContent(text=BUDGET_ACK)],
                    "is_error": False,
                    "details": {"terminal": "max-steps"},
                }

            terminal = _terminal_kind(batch)
            if terminal == "finish":
                # Mirrors the runner: a finish batch is recorded and ends the
                # episode; it is never sent to the adapter (no action in it
                # mutates the environment).
                token.mark_terminal()
                self.end_requested = "finish"
                if self.record is not None:
                    self.record("finish", {"actions": len(batch.actions)})
                return {
                    "content": [TextContent(text=FINISH_ACK)],
                    "is_error": False,
                    "details": {"terminal": "finish"},
                }
            if terminal == "ask_user":
                if self.ask is None:
                    return {
                        "content": [
                            TextContent(
                                text=(
                                    "This episode cannot answer host-owned ask_user "
                                    "actions; continue without asking."
                                )
                            )
                        ],
                        "is_error": True,
                        "details": {"rejection_class": "ask_user-unsupported"},
                    }
                await self.ask(batch)

            observation = token.consume_pending()
            if observation is None:  # pragma: no cover - the lock makes this unreachable
                refusal = _no_pending_refusal(call_id)
                return {
                    "content": refusal.content,
                    "is_error": True,
                    "details": refusal.details,
                }
            result = await self.execute(batch)
            self._steps += 1
            token.record_in_flight(result.observation)
            if self.record is not None:
                self.record("batch", {"batch": batch, "result": result})
            return {
                "content": self.render(result.observation),
                "is_error": False,
                "details": {"receipt": result.receipt.model_dump(mode="json")},
            }


# ---------------------------------------------------------------------------
# Opening the episode's session
# ---------------------------------------------------------------------------


def validate_episode_session_spec(
    *,
    spec: SessionSpec,
) -> SessionSpec:
    """The session spec an episode runs under, from the run's own spec.

    Pinned here rather than at the call site so a comparison between arms
    cannot quietly differ: the full tool surface (``tools=None``), the
    unattended approval posture, and the caller's model/team/profile choices;
    nothing else. The episode's identity rides into the harness as the
    conversation name when the caller left one unset.
    """

    if spec.tools is not None:
        raise SessionArmError(
            "the session arm runs the FULL tool surface; tools= restricts the reach "
            "and would silently change what the arm measures"
        )
    if spec.approvals.mode != "auto":
        raise SessionArmError(
            "the session arm's approvals are fixed at ApprovalPolicy.auto() (exec --yolo "
            "parity); a different policy is a different arm"
        )
    return spec


@dataclass
class EpisodeSession:
    """An open episode session: its live tool inventory and exit handle."""

    session: Any
    roots: SessionRoots
    _context: Any = field(repr=False)
    _unsubscribe: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """The inventory LIVE, never a snapshot taken at open.

        The action server is an MCP server, and MCP discovery is asynchronous:
        its tools merge on settle, after ``open`` returns. A frozen field here
        read 28 names while the session's live inventory already held 29 with
        the action tool in it (measured), and the first request publishes its
        array from the LIVE inventory -- so anything that wants to know what
        the turn can actually call must read it at the moment it asks.
        """

        return session_tool_names(self.session)

    async def aclose(self) -> None:
        """Detach the sink, then exit the SDK's open context."""

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        await self._context.__aexit__(None, None, None)


async def open_episode_session(
    *,
    spec: SessionSpec,
    roots: SessionRoots,
    on_event: Callable[[AgentEvent], Any] | None = None,
    session_opener: Callable[..., Any] | None = None,
) -> EpisodeSession:
    """Open the session an episode runs, subscribing the event sink first.

    The subscription is registered BEFORE the first prompt, because events are
    not replayed: a sink attached after the turn began would miss its open.
    The sink is the RECORD and the token's fold input. Design §6 sketches the
    same job as iteration over ``sdk.events``; subscribing instead keeps the
    fold on the engine's own dispatch order rather than one queue behind it,
    which is what the token's turn-boundary arming requires.
    """

    opener = session_opener or sdk.open_session
    context = opener(spec, roots=roots, mode="own")
    session = await context.__aenter__()
    unsubscribe = session.subscribe(on_event) if on_event is not None else None
    return EpisodeSession(
        session=session,
        roots=roots,
        _context=context,
        _unsubscribe=unsubscribe,
    )


def session_tool_names(session: Any) -> tuple[str, ...]:
    """The session's live tool inventory, by name.

    ``_tools`` first (the live inventory every built session carries),
    ``tools`` second (a host that exposes one), empty last -- the same
    getattr-probe ``session_factory`` itself uses, kept identical so the two
    can never disagree about which list is "the" surface.
    """

    live = getattr(session, "_tools", None) or getattr(session, "tools", None) or ()
    return tuple(tool.name for tool in live)


# ---------------------------------------------------------------------------
# One whole episode, driven as a session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionArmOutcome:
    """What happened in the pilot arm, and where its record lives."""

    status: str
    episode_id: str
    record_root: Path | None
    score: Any | None = None
    steps: int = 0
    terminal_reason: str | None = None
    diagnostic: str | None = None
    rescue_required: bool = False
    rescue_complete: bool | None = None
    tool_names: tuple[str, ...] = ()
    resolved_sources: Mapping[str, str] = field(default_factory=dict)
    duration_ms: int = 0


class _RecordWriter:
    """Append-only JSONL record for one episode; every line is flushed."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


async def run_session_episode(
    *,
    spec: EpisodeSpec,
    config: EpisodeConfig,
    selector: Any,
    roots: SessionRoots,
    scratch_root: Path,
    session_spec: SessionSpec,
    secrets: Sequence[ResolvedSecret] = (),
    display_name: str | None = None,
    max_wall_s: float | None = None,
    launch: Any = AdapterSupervisor.launch,
    rescue: Any = run_rescue,
) -> SessionArmOutcome:
    """Run ONE episode as a session: launch, reset, prompt, score, clean up.

    The order of calls mirrors ``EpisodeRunner`` deliberately, comment for
    comment where the two could drift: prepare is allocation-free, the rescue
    descriptor is re-persisted before ``reset_start`` (the side-effect
    boundary), the artifact root is created by the PARENT, and the score is
    taken after the turn but before cleanup. What differs is the middle: the
    decision loop and the reply channel are replaced by one
    ``sdk.open_session`` turn whose actions arrive as MCP tool calls, and the
    record is the pilot's JSONL rather than the sealed bundle format.
    """

    started_ms = int(time.time() * 1000)
    record_root = config.evidence_root / f"{spec.episode_id}-{SESSION_ARM_ID}"
    record_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = _RecordWriter(record_root / "events.jsonl")

    supervisor: Any = None
    adapter_session: VerifiedAdapterSession | None = None
    bridge: ActionBridge | None = None
    descriptor: RescueDescriptor | None = None
    rescue_required = False
    cleanup_done = False
    plan: CleanupPlan | None = None
    rescue_complete: bool | None = None
    outcome: SessionArmOutcome

    def _mark_rescue_required() -> None:
        nonlocal rescue_required
        rescue_required = True

    try:
        # --- launch + prepare (mirrors EpisodeRunner._launch_and_prepare) ----
        supervisor = launch(selector)
        handshake: Handshake = await supervisor.handshake(timeout=config.handshake_timeout)
        surface: ActionSurface = handshake.metadata.capabilities.action_surface()
        answer_owner = handshake.metadata.capabilities.ask_user_answer_owner
        if answer_owner != "adapter":
            # Preflight rather than a mid-episode refusal: the session arm has
            # no user responder yet, so a host-owned ask cannot be answered at
            # all, and an episode that discovers that at step 7 has already
            # paid for the steps before it.
            raise SessionArmError(
                "the session arm answers asks only through the adapter; this "
                f"adapter's ask_user_answer_owner is {answer_owner!r}"
            )
        verifier = HostVerifier(spec.task_id, spec.episode_id, config.artifact_root)
        adapter_session = VerifiedAdapterSession(
            supervisor,
            verifier,
            rescue_required=_mark_rescue_required,
            answer_owner=answer_owner,
            execution_overhead_seconds_per_action=(config.execution_overhead_seconds_per_action),
        )
        requirements = await adapter_session.inspect_requirements(
            InspectRequirementsParams(), timeout=config.prepare_timeout
        )
        _refuse_undeclared_disclosed_infra(spec, selector, requirements)

        provisional = CleanupPlan(
            episode_id=spec.episode_id,
            actions=(
                CleanupAction(
                    action_id=_PROVISIONAL_CLEANUP_ACTION,
                    kind="close_session",
                    resource_ref=spec.episode_id,
                    timeout_ms=int(config.cleanup_timeout * 1000),
                    max_attempts=2,
                ),
            ),
        )
        descriptor = _persist_descriptor(
            config=config, spec=spec, selector=selector, handshake=handshake, plan=provisional
        )
        adapter_session.mark_rescue_persisted(descriptor.descriptor_id)
        prepared = await adapter_session.prepare(
            PrepareParams(
                operation_id=f"prepare-{spec.episode_id}",
                episode_id=spec.episode_id,
                secret_refs=spec.secret_refs,
                infra_values=spec.infra_values,
            ),
            timeout=config.prepare_timeout,
        )
        plan = prepared.cleanup_plan
        descriptor = _persist_descriptor(
            config=config, spec=spec, selector=selector, handshake=handshake, plan=plan
        )
        adapter_session.mark_rescue_persisted(descriptor.descriptor_id)

        # The parent owns this root (same reason as the runner's: the worker
        # must not get a say in the one directory the parent later opens).
        config.artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        observation = (
            await adapter_session.reset_start(
                ResetStartParams(
                    operation_id=f"reset-{spec.episode_id}",
                    task_id=spec.task_id,
                    episode_id=spec.episode_id,
                    artifact_root=str(config.artifact_root),
                    secrets=tuple(secrets),
                ),
                timeout=config.reset_timeout,
            )
        ).observation
        record.write("reset", {"observation_id": observation.observation_id})

        # --- the action server, declared inside the episode scratch ---------
        # A SHORT, session-unique name, deliberately: ``sun_path`` is bounded
        # (~104 bytes on macOS) and the real arm's scratch is already deep
        # (``<worktree>/scratch-homes/<run-name>``), so a descriptive
        # ``action-bridge-<episode>.sock`` landed at the limit and the guard in
        # ``ActionBridge.start`` refused it (measured: 100+ bytes).
        endpoint = Path(scratch_root) / f"b-{uuid.uuid4().hex[:8]}.sock"
        declaration = declare_action_server(
            config_dir=roots.config_path,
            endpoint=endpoint,
            surface=surface,
            cwd=roots.cwd_path,
        )
        resolved_sources = assert_declaration_resolved(
            cwd=roots.cwd_path,
            config_dir=roots.config_path,
            scratch_root=Path(scratch_root),
        )
        record.write(
            "declaration",
            {
                "server": declaration.server_name,
                "tool": declaration.tool_name,
                "config": str(declaration.config_path),
                "endpoint": str(declaration.endpoint),
                "sources": {name: str(path) for name, path in resolved_sources.items()},
            },
        )

        renderer = ObservationRenderer(config.artifact_root)
        bridge = ActionBridge(
            endpoint=endpoint,
            surface=surface,
            render=renderer.render,
            execute=_make_execute(adapter_session, config, record),
            max_steps=config.max_steps,
            record=lambda kind, payload: record.write(
                "action_" + kind,
                {
                    key: (value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
                    for key, value in payload.items()
                },
            ),
            ask=_make_ask(adapter_session, config, answer_owner),
        )
        bridge.arm(observation)
        await bridge.start()

        # --- the session turn (the engagement this module exists for) -------
        episode_session = validate_episode_session_spec(spec=session_spec)
        if episode_session.name is None and display_name:
            episode_session = replace(episode_session, name=display_name)
        record_errors: list[BaseException] = []

        def _sink(event: AgentEvent) -> None:
            # The sink sits on the engine's own dispatch path, where handler
            # errors are isolated and swallowed -- so a recording failure is
            # captured HERE, reported in the outcome, and the run continues: a
            # torn record stays visible, and a paid episode is not lost to a
            # logging fault.
            try:
                _on_event(record, bridge, event)
            except BaseException as error:  # noqa: BLE001 - see above
                if not record_errors:
                    record_errors.append(error)

        handle = await open_episode_session(spec=episode_session, roots=roots, on_event=_sink)
        try:
            if not await _await_action_tool(handle, declaration, record):
                raise SessionArmError(
                    "the episode's action tool "
                    f"{declaration.tool_name} is not in the session after MCP settle; "
                    "refusing to prompt into a session that cannot act"
                )
            tool_names = handle.tool_names
            record.write("tools", {"names": list(tool_names)})
            text, images = split_prompt_content(renderer.render(observation))
            prompt = PROMPT_HEADER.format(tool_name=declaration.tool_name) + "\n" + text
            wall_timer: asyncio.TimerHandle | None = None
            if max_wall_s is not None:
                # The wall bound is a KILL switch, not a cancellation of the
                # await: abort() ends the turn through the normal stop path,
                # so the record keeps the shape every other end has. The timer
                # is cancelled the moment the turn ends on its own.
                wall_timer = asyncio.get_running_loop().call_later(
                    max_wall_s, handle.session.abort, "episode wall budget"
                )
            try:
                await handle.session.prompt(prompt, images=images or None)
            finally:
                if wall_timer is not None:
                    wall_timer.cancel()
        finally:
            await handle.aclose()

        steps = bridge.steps
        terminal_reason = bridge.end_requested
        if terminal_reason == "finish":
            status = "completed"
        elif terminal_reason == "max-steps" or steps >= config.max_steps:
            status = "truncated"
            terminal_reason = terminal_reason or "max-steps"
        else:
            # The turn ended with no terminal batch. The runner calls the
            # close-equivalents of this an agent stop; the state reached is
            # still scored, exactly as a truncation is.
            status = "agent_stop"

        # --- score, then cleanup, then close (mirrors _close_out) -----------
        score: Any = None
        score_error: BaseException | None = None
        try:
            score = (
                await adapter_session.score(
                    ScoreParams(
                        operation_id=f"score-{spec.episode_id}",
                        episode_id=spec.episode_id,
                    ),
                    timeout=config.score_timeout,
                )
            ).score
            (record_root / "score.json").write_text(
                json.dumps(score.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except BaseException as error:
            # ADAPTER CONTRACT: score() returns a SCORED artifact or raises,
            # and a raise is not a reason to skip cleanup -- the worker is
            # still holding the resource. The failure is recorded here and
            # turned into the outcome by the handler below.
            score_error = error
            record.write(
                "error",
                {"scope": "score", "diagnostic": f"{type(error).__name__}: {error}"},
            )
        receipts: tuple[Any, ...] = ()
        try:
            receipts = await adapter_session.cleanup(
                CleanupParams(
                    operation_id=f"cleanup-{spec.episode_id}",
                    cleanup_plan=plan,
                    action_ids=tuple(action.action_id for action in plan.actions),
                ),
                timeout=config.cleanup_timeout,
            )
        except BaseException as error:
            record.write(
                "error",
                {"scope": "cleanup", "diagnostic": f"{type(error).__name__}: {error}"},
            )
        cleanup_done = True
        # A dead worker cannot produce evidence, and "attempted" alone is never
        # evidence of cleanup -- ``aggregate_cleanup`` is the one statement of
        # that rule (exactly one receipt per declared action; only ``succeeded``
        # and ``not_needed`` aggregate clean).
        rescue_required = rescue_required or _cleanup_forces_rescue(plan, receipts)
        if rescue_required and descriptor is not None:
            rescue_complete = await _attempt_rescue(
                descriptor=descriptor, config=config, secrets=secrets, rescue=rescue
            )
        elif descriptor is not None:
            _discard_descriptor(config)
        if score_error is not None:
            raise score_error
        if record_errors:
            raise SessionArmError(f"the record sink failed: {record_errors[0]!r}")
        outcome = SessionArmOutcome(
            status=status,
            episode_id=spec.episode_id,
            record_root=record_root,
            score=score,
            steps=steps,
            terminal_reason=terminal_reason,
            rescue_required=rescue_required,
            rescue_complete=rescue_complete,
            tool_names=tool_names,
            resolved_sources=resolved_sources,
            duration_ms=int(time.time() * 1000) - started_ms,
        )
    except BaseException as error:
        # Every failure still tears the worker down; a leaked cloud resource is
        # a failed episode regardless of what the model did. Cleanup is
        # attempted even when the failure landed mid-turn -- the runner's
        # failure path runs the same close-out for the same reason -- and a
        # cleanup that cannot complete forces the rescue path.
        if adapter_session is not None and plan is not None and not cleanup_done:
            try:
                receipts = await adapter_session.cleanup(
                    CleanupParams(
                        operation_id=f"cleanup-{spec.episode_id}",
                        cleanup_plan=plan,
                        action_ids=tuple(action.action_id for action in plan.actions),
                    ),
                    timeout=config.cleanup_timeout,
                )
                rescue_required = rescue_required or _cleanup_forces_rescue(plan, receipts)
            except BaseException:
                rescue_required = True
        if rescue_required and descriptor is not None:
            rescue_complete = await _attempt_rescue(
                descriptor=descriptor, config=config, secrets=secrets, rescue=rescue
            )
        outcome = SessionArmOutcome(
            status="failed_pre_bundle" if adapter_session is None else "failed",
            episode_id=spec.episode_id,
            record_root=record_root,
            diagnostic=f"{type(error).__name__}: {error}",
            rescue_required=rescue_required,
            rescue_complete=rescue_complete,
            duration_ms=int(time.time() * 1000) - started_ms,
        )
        record.write("error", {"diagnostic": outcome.diagnostic})
    finally:
        if bridge is not None:
            await bridge.stop()
        await _close_adapter_session(adapter_session, supervisor, spec, config, rescue_required)
        record.close()

    (record_root / "outcome.json").write_text(
        json.dumps(_outcome_json(outcome), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outcome


async def _await_action_tool(
    handle: Any,
    declaration: ActionServerDeclaration,
    record: "_RecordWriter",
    *,
    timeout_s: float = 30.0,
) -> bool:
    """Hold the prompt until the action tool provably exists in the session.

    MCP discovery is asynchronous BY DESIGN: a server deferred past the 250 ms
    startup gate connects on a background continuation and its tools merge on
    settle (``session_factory`` -> ``apply_preload``). The episode's FIRST
    request is the one request that must not race that settle -- it is the
    request whose tool array carries the action tool -- and a session that
    prompts without it answers every model call with "Tool not found: <action
    tool>" (measured: the whole pilot turn failed exactly that way when the
    prompt went out ~10 ms after open). So the arm waits, bounded, and REFUSES
    to prompt when the tool never arrives: absence is a loud, recorded failure,
    never a silent episode that cannot act.
    """

    manager = getattr(handle.session, "mcp_manager", None)
    settled: bool | None = None
    if manager is not None:
        try:
            settled = await manager.wait_settled(timeout_s)
        except Exception:  # noqa: BLE001 - a reduced manager must not break the arm
            settled = None
    present = declaration.tool_name in handle.tool_names
    startup = getattr(handle.session, "mcp_startup", None)
    record.write(
        "mcp_settle",
        {
            "settled": settled,
            "action_tool_present": present,
            "configured": list(getattr(startup, "configured", ()) or ()),
            "connected": list(getattr(startup, "connected", ()) or ()),
            "failures": {
                name: str(message)
                for name, message in (getattr(startup, "failures", None) or {}).items()
            },
        },
    )
    return present


def _cleanup_forces_rescue(plan: CleanupPlan, receipts: Sequence[Any]) -> bool:
    """Whether cleanup leaves the worker unconfirmed: the runner's own rule.

    ``aggregate_cleanup`` requires exactly one receipt per declared action and
    only aggregates ``succeeded``/``not_needed`` as clean; an empty or
    incomplete receipt set therefore forces the rescue path, exactly as the
    runner's ``_run_cleanup`` does (which mints incomplete receipts for a dead
    worker so this same aggregation turns red rather than reading as success).
    """

    if not receipts:
        return True
    try:
        return bool(aggregate_cleanup(plan, tuple(receipts)).rescue_required)
    except ValueError:
        return True


def _on_event(record: _RecordWriter, bridge: ActionBridge, event: AgentEvent) -> None:
    """One sink for the whole session stream: record it, fold it, count it."""

    bridge.fold(event)
    record.write("agent_event", {"event": printable_event(event)})


def _make_execute(
    adapter_session: VerifiedAdapterSession, config: EpisodeConfig, record: _RecordWriter
) -> ExecuteBatch:
    """The execution seam: one batch, with the runner's read-back recovery.

    Copied in ORDER from ``EpisodeRunner._execute_with_observation_recovery``,
    which is the one place the contract "retry only what the adapter declared
    committed" is implemented; the two must not drift, so this mirrors its
    attempt loop, its fresh operation ids, and its decision that any other
    failure propagates.
    """

    async def execute(batch: ActionBatch) -> Any:
        from local_operator.evaluation.runner.episode import _adapter_batch_id

        operation_id = f"exec-{uuid.uuid4().hex[:12]}"
        params = ExecuteParams(
            operation_id=operation_id,
            action_batch=batch,
            action_batch_id=_adapter_batch_id(batch),
        )
        attempts = max(0, config.observation_retry_attempts)
        for attempt in range(attempts + 1):
            try:
                if attempt == 0:
                    if config.execution_overhead_seconds_per_action:
                        return await adapter_session.execute(
                            params,
                            timeout=config.step_timeout,
                            execution_overhead_seconds_per_action=(
                                config.execution_overhead_seconds_per_action
                            ),
                        )
                    return await adapter_session.execute(params, timeout=config.step_timeout)
                return await adapter_session.resume_observation(
                    params.model_copy(update={"operation_id": f"{operation_id}-obs{attempt}"}),
                    timeout=config.step_timeout,
                )
            except Exception as error:
                if attempt == attempts or not _observation_phase_failure(error):
                    raise
                record.write(
                    "observation_retry",
                    {
                        "attempt": attempt + 1,
                        "attempts": attempts,
                        "detail": str(error)[:500],
                    },
                )
                await asyncio.sleep(config.observation_retry_delay)

    return execute


def _make_ask(
    adapter_session: VerifiedAdapterSession, config: EpisodeConfig, answer_owner: str
) -> Callable[[ActionBatch], Awaitable[str | None]] | None:
    """The ask seam, adapter-owned answers only (the pilot arm's posture)."""

    if answer_owner != "adapter":
        # Host-owned answers need a responder surface the pilot arm does not
        # carry; refusing here turns a silent "the model asked and nobody could
        # answer" into a named precondition. The reply-channel arm is the one
        # that answers these today.
        return None

    async def ask(batch: ActionBatch) -> str | None:
        from local_operator.evaluation.runner.episode import _ask_prompt

        ask_id, prompt = _ask_prompt(batch)
        begin = AskUserExchangeParams(
            operation_id=f"ask-begin-{ask_id}",
            episode_id=batch.episode_id,
            ask_id=ask_id,
            prompt=prompt,
        )
        adapter_session.begin_ask(begin)
        result = await adapter_session.finish_ask(begin, timeout=config.step_timeout)
        return result.answer

    return ask


def _refuse_undeclared_disclosed_infra(
    spec: EpisodeSpec, selector: Any, requirements: RequirementsResult
) -> None:
    """Refuse a disclosed infra value the adapter build does not understand.

    Same rule as the runner's (``EpisodeRunner._refuse_undeclared_disclosed_infra``):
    an override the adapter would silently drop is worse than a failed episode,
    because the record would then disagree with the run.
    """

    from local_operator.evaluation.runner.episode import _DISCLOSED_INFRA_VALUES

    declared = {requirement.name for requirement in requirements.requirements}
    supplied = {value.name for value in spec.infra_values}
    undeclared = sorted((supplied & _DISCLOSED_INFRA_VALUES) - declared)
    if undeclared:
        raise UndeclaredDisclosedInfra(
            f"adapter {selector.adapter_id!r} version {selector.version!r} does not "
            f"declare {undeclared}, so the value would be silently ignored while the "
            "recorded run showed it applied; rebuild the adapter workspace and selector, "
            "or drop the value"
        )


def _persist_descriptor(
    *,
    config: EpisodeConfig,
    spec: EpisodeSpec,
    selector: Any,
    handshake: Handshake,
    plan: CleanupPlan,
) -> RescueDescriptor:
    """Persist the rescue descriptor exactly as the runner does, twice."""

    descriptor = RescueDescriptor(
        schema_version=ADAPTER_SCHEMA_VERSION,
        selector=selector,
        handshake=handshake,
        episode_id=spec.episode_id,
        cleanup_plan=plan,
        secret_refs=spec.secret_refs,
        infra_values=spec.infra_values,
        artifact_root=str(config.artifact_root),
    )
    persist_rescue(config.rescue_root, descriptor)
    return descriptor


def _discard_descriptor(config: EpisodeConfig) -> None:
    try:
        discard_rescue(config.rescue_root)
    except OSError:
        pass


async def _attempt_rescue(
    *,
    descriptor: RescueDescriptor,
    config: EpisodeConfig,
    secrets: Sequence[ResolvedSecret],
    rescue: Any,
) -> bool:
    try:
        aggregate = await rescue(descriptor, secrets=tuple(secrets))
    except BaseException:
        return False
    complete = bool(aggregate.complete)
    if complete:
        _discard_descriptor(config)
    return complete


async def _close_adapter_session(
    adapter_session: VerifiedAdapterSession | None,
    supervisor: Any,
    spec: EpisodeSpec,
    config: EpisodeConfig,
    rescue_required: bool,
) -> None:
    """Mirror ``EpisodeRunner._close_session``: close, then terminate."""

    if adapter_session is not None and not rescue_required:
        try:
            await adapter_session.close(
                CloseParams(
                    operation_id=f"close-{spec.episode_id}",
                    episode_id=spec.episode_id,
                ),
                timeout=config.cleanup_timeout,
            )
        except BaseException:
            pass
    if supervisor is not None:
        try:
            await supervisor.terminate()
        except BaseException:
            pass


def _outcome_json(outcome: SessionArmOutcome) -> dict[str, Any]:
    score = outcome.score
    score_payload: Any = None
    if score is not None:
        score_payload = score.model_dump(mode="json") if hasattr(score, "model_dump") else score
    return {
        "arm": SESSION_ARM_ID,
        "status": outcome.status,
        "episode_id": outcome.episode_id,
        "record_root": str(outcome.record_root) if outcome.record_root else None,
        "score": score_payload,
        "steps": outcome.steps,
        "terminal_reason": outcome.terminal_reason,
        "diagnostic": outcome.diagnostic,
        "rescue_required": outcome.rescue_required,
        "rescue_complete": outcome.rescue_complete,
        "tool_names": list(outcome.tool_names),
        "resolved_sources": {name: str(path) for name, path in outcome.resolved_sources.items()},
        "duration_ms": outcome.duration_ms,
    }
