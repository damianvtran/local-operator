"""Core harness types — the binding contract between all rewrite streams.

This module defines the ``packages/agent`` type surface. It deliberately knows
NOTHING about sessions, providers, persistence, skills, or UI. Every other
stream programs against these types:

- ``Message`` / ``CustomMessage`` — LLM-visible conversation entries.
- ``ToolCall`` / ``ToolResult`` / ``AgentTool`` — the tool protocol.
- ``AgentEvent`` — the ONLY boundary between engine and UI (TUI, print mode,
  server websockets all subscribe to these).
- ``LoopConfig`` — the callback bundle injected into the loop. The loop never
  imports session or provider code; everything is a callback here.
- ``ModelSpec`` / ``ChatRequest`` / ``StreamEvent`` — the provider wire
  contract implemented by ``local_operator.providers.clients``.

Design notes carried over from the reference engine:

- Messages carry an optional ``provider_payload`` for provider-native replay
  data (e.g. OpenAI Responses ids, Anthropic encrypted thinking). It rides
  through history untouched and is consumed only by wire clients.
- ``CustomMessage`` is the extension point for host-authored entries
  (compaction summaries, skill prompts, wake deliveries). It renders to an
  LLM-visible message via the session's ``convert_to_llm`` and NEVER goes to
  a provider raw.
- Aside commit/discard: the reference engine attaches symbols to message
  objects; here they
  are explicit optional callables excluded from serialization
  (``compare=False``), invoked by the loop when an aside message is actually
  injected (commit) or dropped as stale (discard).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Protocol,
    Sequence,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

# TypeVar comes from typing_extensions, NOT typing: the ``default=`` parameter
# below is PEP 696, which landed in typing only in 3.13, while this package
# supports 3.12. On 3.12 ``typing.TypeVar(default=...)`` raises TypeError at
# import time, which surfaces as an undiagnosable "preflight failed" on every
# command. typing_extensions backports it and is a declared dependency for
# exactly this reason (it also arrives with pydantic).
from typing_extensions import TypeVar

# One-way, in-package dependencies: ``approval`` (the gate's type plus the one
# arity resolver) and ``wake`` (pure schedule data plus a timer) each import
# nothing else from the harness, so naming their types here cannot cycle. They
# buy the approval-gate and wake-scheduler contracts on ``ToolContext`` real
# types instead of ``Any``.
from local_operator.harness.approval import ApprovalGate
from local_operator.harness.wake import WakeSchedule

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RenderedStreamError(Exception):
    """A stream failure whose ``str()`` is the whole story for the user.

    Raised by wire clients for provider responses (``HTTP 400: ...``) as
    opposed to defects. The loop catches both, but only the latter is worth a
    traceback: a handled provider answer that the UI already prints as one
    clean line does not also need forty lines of stack painted over the
    interface. Lives here rather than in ``providers`` because the harness must
    not import the provider layer — the dependency only runs the other way.
    """


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    """A plain text block inside a message."""

    model_config = ConfigDict(frozen=False)

    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    """An image block. ``data`` is base64-encoded bytes of ``mime_type``."""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = "image/png"


Content = TextContent | ImageContent


# ---------------------------------------------------------------------------
# Tool calls and results
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """One requested tool invocation as emitted by the model."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Raw JSON argument string when the provider gives us one; wire clients
    # replay this verbatim for providers that require it.
    raw_arguments: str | None = None


class ToolResult(BaseModel):
    """The outcome of executing one ``ToolCall``.

    ``is_error`` marks a NON-throwing failure that should go back to the model
    as a normal tool result (never raise into the loop for model-recoverable
    errors). ``useless`` flags a contextually worthless result (zero-match
    search, timed-out wait) that compaction may elide once consumed; it must
    never be set together with ``is_error`` (errors win).
    """

    tool_call_id: str
    tool_name: str = ""
    content: list[Content] = Field(default_factory=list)
    # Structured payload for renderers, logs and compaction pruning. Never
    # serialized to providers; always a JSON-ish mapping so consumers can
    # index it without probing the value's shape first.
    details: dict[str, Any] | None = None
    # Active wall time is captured beside execution, not reconstructed by a
    # replaying surface whose clock starts when it paints the historical row.
    duration_s: float | None = None
    is_error: bool = False
    useless: bool = False

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

MessageRole = Literal["user", "assistant", "tool"]


class Message(BaseModel):
    """One LLM-visible message.

    Identity matters: compaction memoizes token estimates per message object.
    Mutating a message in place (pruning, streaming finalize) MUST call
    ``invalidate_message_cache`` on the compaction cache — see
    ``local_operator.compaction``.
    """

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: list[Content] = Field(default_factory=list)
    # assistant only: requested tool calls for this turn
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # tool only: which call this result answers
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    stop_reason: str | None = None  # stop | length | toolUse | refusal | error | aborted
    usage: "Usage | None" = None
    # Provider-native replay payload (opaque to the harness). NOTE: the loop
    # stores harness bookkeeping under ``provider_payload["details"]`` (tool
    # result metadata for compaction) — wire clients MUST NOT replay that key
    # to providers; it is not provider data.
    provider_payload: dict[str, Any] | None = None
    # Stable id for transcript entries and cache memoization.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @staticmethod
    def user(text: str, images: "Sequence[ImageContent] | None" = None, **extra: Any) -> "Message":
        """A user turn, optionally carrying attachments.

        Images follow the text rather than leading it: the prompt says what to
        do with them, and a model reading the instruction first knows what it
        is looking for. Empty text still yields a text block, so a message that
        is nothing but a pasted screenshot keeps the shape every provider
        serializer expects.
        """
        content: list[Content] = [TextContent(text=text)]
        content.extend(images or ())
        return Message(role="user", content=content, **extra)

    @staticmethod
    def assistant(text: str = "", **extra: Any) -> "Message":
        return Message(role="assistant", content=[TextContent(text=text)] if text else [], **extra)

    @staticmethod
    def tool_result(result: ToolResult) -> "Message":
        return Message(
            role="tool",
            content=list(result.content),
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            is_error=result.is_error,
        )


class Usage(BaseModel):
    """Token accounting reported by a provider (or estimated locally)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # The TTL split of ``cache_write_tokens`` when the provider reports one
    # (Anthropic's ``usage.cache_creation.ephemeral_5m_input_tokens`` /
    # ``ephemeral_1h_input_tokens``). SUBSETS of ``cache_write_tokens``, never
    # added on top — the docs state ``cache_creation_input_tokens`` equals their
    # sum, and every existing consumer of ``cache_write_tokens`` stays correct.
    # They exist because the two TTLs are priced differently (1.25× vs 2× base):
    # once the Anthropic client starts writing 1h entries on large contexts,
    # analytics needs to tell the two apart to know whether the trade paid off.
    # Both stay 0 on providers without a TTL split, and on an Anthropic response
    # that omits the ``cache_creation`` object (older API versions).
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    context_tokens: int | None = None  # provider-reported full context size if given
    # The reasoning/thinking slice of ``output_tokens`` when the provider
    # breaks it out (OpenAI ``output_tokens_details.reasoning_tokens``). Kept
    # as a SUBSET of ``output_tokens`` — never added on top — so callers that
    # only read ``output_tokens`` stay correct, and the analytics recorder can
    # split output into thinking vs generation (``output_tokens`` minus this).
    # Anthropic bills thinking inside ``output_tokens`` without a separate
    # count, so this stays 0 there and the whole output reads as generation,
    # which is the honest thing to report when the wire does not separate it.
    reasoning_tokens: int = 0
    # Provider-reported dollar cost for this one request, when the provider
    # precomputes billing (OpenRouter's ``usage.cost``). This is the ground truth
    # a caller must prefer over any token×rate reconstruction: the provider has
    # already applied per-route pricing, reasoning-token splits, cache discounts
    # and time/value overrides that a single flat table price cannot express.
    # ``None`` means "not reported" and is distinct from a real ``0.0`` (a call
    # the provider billed as free) — the same three-way split the TUI's
    # ``None``-vs-``$0.0000`` contract already draws.
    usd_cost: float | None = None
    # Aggregate-only copies of the calls behind this usage. Token buckets alone
    # cannot preserve which calls carried authoritative provider receipts and
    # which still need a table estimate; keeping the components lets money be
    # priced call-by-call without billing the aggregate tokens a second time.
    # Wire/provider usages leave this empty. Parent turns and child ledgers fill
    # it while folding message usages together.
    cost_components: list["Usage"] = Field(default_factory=list)
    # Serving identity, stamped by the failover layer from the on-the-wire
    # ``ChatRequest`` (the spec that actually went out). The analytics recorder
    # used to read ``request.model`` off the ORIGINAL ChatRequest, which still
    # names the session primary after ``stream_with_failover`` rewrites the
    # request to a fallback — every Grok call then landed under
    # ``anthropic/claude-opus-4-8`` and was priced at Opus rates. These fields
    # are the honest channel: a primary success, an isolated/naming call
    # (``route_state`` is None), and a mid-turn failover all carry the spec
    # that served THIS attempt. ``None`` means "not stamped"; the recorder
    # then falls back to ``request.model``.
    provider: str | None = None
    model_id: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CustomMessage(BaseModel):
    """Host-authored transcript entry that renders into LLM context via
    ``convert_to_llm``. Subclass-free extension: ``custom_type`` discriminates
    (``"compaction_summary"``, ``"skill_prompt"``, ``"wake_prompt"``,
    ``"handoff"``, ...), ``details`` carries the typed payload.

    ``id`` is the stable transcript entry id: the transcript persists it
    verbatim (never mints a new one) and ``convert_to_llm`` must carry it
    onto the rendered message, so ``first_kept_entry_id`` can reference a
    rendered custom entry and replay still finds it.

    The two callables are aside commit/discard hooks (see module docstring);
    they are never serialized. ``on_commit`` fires when the message is
    actually injected into context; ``on_discard`` fires when the aside is
    dropped as stale at injection time.
    """

    model_config = ConfigDict(extra="allow")

    custom_type: str
    attribution: Literal["user", "agent", "system"] = "system"
    details: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    on_commit: Callable[[], None] | None = Field(
        default=None, exclude=True, json_schema_extra={"compare": False}
    )
    on_discard: Callable[[], None] | None = Field(
        default=None, exclude=True, json_schema_extra={"compare": False}
    )


class StaleAside:
    """Returned by an aside thunk when its payload is stale at injection
    time. Carries the originating :class:`CustomMessage` so the loop can fire
    its ``on_discard`` hook; the message itself is never injected. (A plain
    ``None`` thunk result is dropped silently — producers that need the
    discard receipt must return this instead.)"""

    __slots__ = ("message",)

    def __init__(self, message: CustomMessage) -> None:
        self.message = message


AgentMessage = Message | CustomMessage

#: What evaluating an aside yields: a live message to inject, a stale-receipt
#: so the producer's ``on_discard`` still fires, or nothing at all.
AsideResult = AgentMessage | StaleAside | None
#: A queued aside is either a ready message or a thunk evaluated at the
#: injection boundary. The thunk form is the point: a payload that went stale
#: while the turn ran can withdraw itself instead of being injected.
Aside = AsideResult | Callable[[], AsideResult]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class AgentToolUpdate(BaseModel):
    """A partial result streamed from a running tool."""

    content: list[Content] = Field(default_factory=list)
    # Same contract as ``ToolResult.details``: a mapping or nothing.
    details: dict[str, Any] | None = None


# --- capability contracts carried on ToolContext ---------------------------
#
# These are Protocols rather than concrete classes for two reasons: the
# harness must not import the session/app layers that own the
# implementations, and hosts (and tests) legitimately supply their own
# stand-ins. They are ``runtime_checkable`` because pydantic validates an
# arbitrary-type field with ``isinstance``.


@runtime_checkable
class WakeSchedulerProtocol(Protocol):
    """The slice of a wake scheduler the ``wake`` tool drives.

    Implemented by :class:`local_operator.harness.wake.WakeScheduler`. The
    tool only ever reads the current list and writes a replacement, so the
    arming/timer half of the scheduler stays out of the contract.
    """

    @property
    def schedules(self) -> Sequence[WakeSchedule]: ...

    async def update(self, schedules: list[WakeSchedule]) -> None: ...


@runtime_checkable
class PeerArrivalProtocol(Protocol):
    """A wakeable signal that an inbound peer message (``lop send``) landed.

    Exists so a blocking tool can park on peer arrival WITHOUT importing the
    session. ``wait`` is the only consumer today and the only one that should
    be: it is a read-only sleep, so waking it early costs nothing, and the
    message it wakes for is already in the session's journal by then. Do NOT
    reuse this to preempt a MUTATING tool — mailbox delivery is
    non-interrupting by contract (``guides/peer-messaging/GUIDE.md``), and
    cancelling a ``bash`` mid-side-effect to hand the model "skipped" is a
    price only a human pressing Esc gets to charge.

    Threading: the session's implementation sets the event on the loop that
    owns the session, because every registrant path hops there first
    (``mobile/tui_handle.py``, ``mobile/owned.py`` both use
    ``run_coroutine_threadsafe``). A future caller that invokes
    ``receive_peer_message`` from its own thread WITHOUT that hop would need
    ``loop.call_soon_threadsafe`` — ``asyncio.Event.set`` is not thread-safe.
    """

    def event(self) -> asyncio.Event:
        """An Event set on each inbound peer message.

        Never cleared by the producer. Consumers snapshot :meth:`count` before
        parking and compare after waking, which is what makes a message that
        arrives BETWEEN two parks impossible to miss.
        """
        ...

    def count(self) -> int:
        """Monotonic count of peer messages delivered to this session."""
        ...


@runtime_checkable
class VariableStoreProtocol(Protocol):
    """The slice of a variable store the variables tools read.

    Implemented by :class:`local_operator.variables.VariableStore`. Listing
    yields names only — values are pulled one at a time — so the store's
    denylist stays the single gate on what the model can see.
    """

    def names(self) -> list[str]: ...

    def get(self, name: str) -> str | None: ...

    def read(self, name: str) -> str:
        """Resolve ``name`` or raise ``KeyError`` when unknown or denied."""
        ...

    def store_credential(
        self, raw_key: str, value: str, source: Literal["command", "ask"] = "command"
    ) -> Any:
        """Capture a session-only secret. See :class:`local_operator.variables.VariableStore`."""
        ...

    def forget_credential(self, raw_key: str) -> bool: ...

    def clear_credentials(self) -> int: ...

    def credential_names(self) -> list[str]: ...

    def list_credentials(self) -> list[Any]: ...

    def credential_env(self) -> dict[str, str]: ...

    def redact(self, text: str) -> str: ...


@runtime_checkable
class BrowserSurfaceProtocol(Protocol):
    """Mutable handle to the host browser surface a session has open.

    The browser tool records the handle here on ``open`` and every later
    action drives that surface instead of leaking a fresh one per call.
    """

    surface_id: str


class BrowserSurface:
    """The concrete :class:`BrowserSurfaceProtocol` a HOST owns.

    Deliberately host-owned rather than created by the tool on demand: the
    session rebuilds its :class:`ToolContext` at the start of EVERY turn, so a
    handle the tool stashed on the context survived only the turn that opened
    it. That broke the ordinary shape of browsing ("open X" then, next
    message, "click Y") and stranded a cmux tab per turn that nothing could
    close. Injected like ``wake_scheduler``, the surface outlives the context
    and session teardown can close it.

    Lives here beside the protocol, and holds no cmux knowledge, so a host can
    own one without importing the tool layer.
    """

    __slots__ = ("surface_id",)

    def __init__(self) -> None:
        self.surface_id = ""


@runtime_checkable
class JobManagerProtocol(Protocol):
    """The slice of ``harness.jobs.AsyncJobManager`` the tools drive.

    Declared HERE rather than importing the manager because ``harness.jobs``
    imports THIS module: annotating ``ToolContext.jobs`` with the concrete
    class would be an import cycle. Exposes exactly what ``wait``/``job``
    and cancellation need (get, list, cancel) and nothing more — spawning
    reaches the manager through session-installed launcher closures, never
    through this surface, so the tools cannot touch the manager's
    registration or delivery internals. Return types stay ``Any`` for the
    same edge reason: the ``AsyncJob`` row type cannot be named on this side
    of the cycle.
    """

    def get(self, job_id: str, *, owner_id: str | None = None) -> Any: ...

    def list(self, *, owner_id: str | None = None) -> list[Any]: ...

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool: ...

    # Three methods are deliberately NOT declared here: ``settled_event``
    # (event-driven ``wait``) and ``append_output``/``read_output`` (the live
    # output channel behind ``jobs(op="peek")``).
    #
    # This Protocol is ``runtime_checkable`` and ``ToolContext.jobs`` validates
    # against it, so every method added becomes mandatory for every existing
    # implementation — a host, an embedder's manager, or a test double written
    # against the older surface would stop validating the moment it shipped.
    # Adding the output pair here really did break eight such doubles outright.
    #
    # Each caller therefore probes with ``getattr`` and degrades: ``wait`` falls
    # back to polling (``tools.builtin._await_any_settled``), and peek reports
    # "this manager records no live output" (``tools.builtin._peek_job``, and
    # bash's output mirror). That costs one branch and keeps the capability
    # opt-in for third-party managers rather than breaking them.


@runtime_checkable
class SubagentLauncher(Protocol):
    """Spawn a one-shot child session on the session's job manager.

    ``agent`` selects the tier: ``"task"`` is the full child, ``"scout"`` a
    read-only research child (its tool inventory is filtered to retrieval that
    changes nothing — local lookups plus web search/fetch — never to edits or
    execution). ``effort`` routes to a configured model tier (``lo``/``med``/
    ``hi`` in ``values.subagents.models``); None keeps the parent's model.
    """

    def __call__(
        self,
        label: str,
        prompt: str,
        *,
        agent: str = "task",
        effort: str | None = None,
    ) -> str: ...


# Both models are NESTED in the ask tool's JSON schema, so their docstrings ride
# in the tools array of every request. The reasoning therefore lives in comments
# here and the docstrings stay one line — the same reason the other tool params
# models in `tools/builtin.py` carry no prose.
#
# `description` exists because the labels a model writes are short by necessity:
# a row reading `Escalate it` cannot say what escalating costs, and the prose
# version of this surface (lettered options printed into the transcript) always
# carried that second clause.
class AskOption(BaseModel):
    """One selectable answer on an ``ask`` question."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, description="The answer, as one short line.")
    description: str = Field(
        default="", description="One line under the label: what choosing it means."
    )


# Two options is the FLOOR, not a style preference: a one-option question is an
# announcement, and rendering it as a picker asks the user to ratify a decision
# that has already been made. A model with nothing to choose between should say
# so in prose instead.
#
# `recommended` indexes `options` and is validated rather than clamped: a silent
# clamp would preselect and visibly endorse a DIFFERENT option than the model
# meant to.
class AskQuestion(BaseModel):
    """One question the ``ask`` tool puts to the user."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description="Short stable key for this question; the answer is reported under it.",
    )
    question: str = Field(min_length=1, description="The question, as one sentence.")
    options: list[AskOption] = Field(
        default_factory=list,
        description="Answers to pick from. Required unless secret is true.",
    )
    multi: bool = Field(
        default=False, description="True lets the user pick several options instead of one."
    )
    recommended: int | None = Field(
        default=None,
        description="0-based index of the option you recommend; it is preselected and marked.",
    )
    secret: bool = Field(
        default=False,
        description=(
            "Request a credential instead of a choice. The answer is stored in "
            "session memory under this question's id and only the key name is returned."
        ),
    )

    @model_validator(mode="after")
    def _shape(self) -> "AskQuestion":
        if self.secret:
            # A secret question is a masked paste, not a picker. Options and
            # multi-select have no meaning there, and a recommended option
            # would preselect a choice nobody can see. The id IS the
            # credential key, so it has to survive normalize_credential_key.
            if self.options:
                raise ValueError("a secret question has no options; the answer is a pasted value")
            if self.multi:
                raise ValueError("a secret question cannot be multi-select")
            if self.recommended is not None:
                raise ValueError("a secret question has no options to recommend")
            from local_operator.variables import normalize_credential_key

            if normalize_credential_key(self.id) is None:
                raise ValueError(
                    "a secret question's id must be a usable credential key "
                    "(letters, digits, underscores)"
                )
            return self
        if len(self.options) < 2:
            raise ValueError("at least two answers to pick from")
        if self.recommended is not None and not 0 <= self.recommended < len(self.options):
            raise ValueError(
                f"recommended must index options (0..{len(self.options) - 1}), "
                f"got {self.recommended}"
            )
        return self


#: The host's interactive-question hook: put these questions to the user and
#: return ``question id -> the strings they chose``. ``None`` means the user
#: answered NOTHING (escaped out), which is a legitimate outcome rather than a
#: failure — the ask tool reports it as one so the model falls back to its own
#: recommendation instead of retrying a question that will be refused again.
#:
#: A list of strings even for a single-select question, and a FREE string
#: rather than an option index: the picker's "Other" row hands back text that
#: was never in ``options``, which an index cannot express.
AskUserFn = Callable[[list[AskQuestion]], Awaitable[dict[str, list[str]] | None]]


class ToolContext(BaseModel):
    """Minimal host-provided context handed to tool execution.

    Kept tiny on purpose (a 100-field monolithic session object is a symptom
    of a 9000-line session class); grow by demand. ``extra="allow"`` remains
    so a host can stash something bespoke, but every capability the built-in
    tools look for is DECLARED below — a tool must not have to probe for an
    undeclared attribute to find out what its host supports.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    cwd: str = "."
    session_id: str = ""
    # Human-readable title is display metadata only. Security-sensitive tools
    # must continue using ``session_id`` for identity and authorization.
    session_name: str = ""
    agent_id: str = ""
    # Which BACKGROUND JOB this execution belongs to, so a host can scope an
    # approval decision to the work that provoked it instead of to every
    # request that follows. ``None`` is the foreground: the session's own turn.
    # Not telemetry and not an identity — ``session_id``/``agent_id`` already
    # say who is asking; this says on whose behalf, which is the part a host
    # cannot otherwise recover. Live failure it exists for: a subagent running
    # past the end of its parent's turn inherited that turn's approval state
    # and had its tools denied with no prompt shown to anyone.
    job_id: str | None = None
    has_ui: bool = False
    # Resolver hook for lazy internal URLs (``skill://`` and ``guide://``);
    # returns content or None when the URL is not handled. Installed by session.
    resolve_internal_url: Callable[[str], str | None] | None = None
    # Approval callback: returns True when the user approved. Tools with an
    # approval tier call this before mutating side effects. Two shapes are
    # accepted (see harness/approval.py); call it through ``ask_approval``
    # rather than directly, which is what picks the one this host wrote.
    request_approval: ApprovalGate | None = None
    # Session-named variables behind the list_variables / read_variable tools.
    # Values are never baked into the prompt; the model lists names and reads
    # single values on demand. ``None`` degrades those tools to the process
    # environment only.
    variables: VariableStoreProtocol | None = None
    # Startup snapshot used only by the web_search createIf gate. Execution
    # re-reads config so provider toggles apply to the next call without
    # rebuilding the session; the master on/off switch removes the advertised
    # tool on reload.
    web_search_settings: dict[str, Any] | None = None
    # Startup snapshot used only by the web_fetch createIf gate, mirroring
    # web_search: execution re-reads config so knobs (TTL, allow_private) apply
    # to the next call without rebuilding the session, while the master on/off
    # switch removes the advertised tool on reload.
    web_fetch_settings: dict[str, Any] | None = None
    # Session-owned capability tools that built-ins may delegate to. This is
    # deliberately a mapping rather than a second MCP client: OAuth transports
    # and reconnect state must remain owned by the one session MCP manager.
    delegated_tools: dict[str, Any] = Field(default_factory=dict)
    # Wake scheduling. ``None`` means the host has no scheduler, and the wake
    # tool is then not advertised at all (createIf) rather than advertised and
    # always failing.
    wake_scheduler: WakeSchedulerProtocol | None = None
    # Durable todo lists keyed by session id. A host that attaches one gets
    # todo state it can persist alongside the transcript; otherwise the tool
    # falls back to a process-local table.
    todos: dict[str, list[dict[str, str]]] | None = None
    # Optional owner hook for canonical full-TUI state. Tools call it only after
    # a successful mutation, never on read-only view or validation failures.
    on_todos_changed: Callable[[], None] | None = None
    # Injected by the HOST (see BrowserSurface), not created by the tool: this
    # context is rebuilt every turn, so a tool-owned handle would not survive
    # to the next one. ``None`` degrades the browser tool to a single-call
    # surface the session can never close.
    browser: BrowserSurfaceProtocol | None = None
    # Launcher for one-shot child sessions run as background jobs (the
    # ``task`` tool). ``(label, prompt) -> job_id``, registering the run as
    # an AsyncJob on the host's job manager. The session installs a closure
    # over its own emit and job manager; ``None`` means the host has no
    # subagent engine and the task tool is then not advertised at all
    # (createIf) rather than advertised and always failing — the same
    # convention ``wake_scheduler`` uses.
    subagent_launcher: "SubagentLauncher | None" = None
    # The user's persistent agent registry (``local_operator.agents``), behind
    # the ``agent`` tool and behind role resolution for ``task(agent=...)``.
    # Typed ``Any`` because that module is heavy (dill, yaml, the whole agent
    # state machine) and importing it here would pull it into every process
    # that merely wants a tool type. ``None`` means the host keeps no registry:
    # the ``agent`` tool is then not advertised (createIf) and delegation falls
    # back to packaged starter profiles.
    agent_registry: Any = None
    # The user's persistent team registry (``local_operator.teams``), behind
    # the ``team`` tool and behind ``/team <name> <request>``. Typed ``Any``
    # for the same reason ``agent_registry`` is: importing that module here
    # would pull yaml persistence into every process that merely wants a
    # tool type. ``None`` means the host keeps no teams: the ``team`` tool
    # is then not advertised (createIf).
    team_registry: Any = None
    # The session's background job manager. Declared as a Protocol because
    # the concrete class lives in ``harness.jobs``, which imports this
    # module (import cycle). The ``wait``/``job`` tools read it; ``None``
    # means no background work is tracked and both tools are then not
    # advertised at all (createIf) rather than advertised and always
    # failing.
    jobs: JobManagerProtocol | None = None
    # Set by the session on each inbound ``lop send`` delivery so a blocking
    # ``wait`` can return early and let the model read its mailbox. Declared
    # rather than probed for, per this class's contract above. ``None`` means
    # the host has no peer surface, and ``wait`` keeps exactly its old three
    # wake sources (job settle / abort / deadline) — this field only ever adds
    # a fourth, it never removes one.
    peer_arrival: PeerArrivalProtocol | None = None
    # The parent↔child messaging surface behind the ``hub`` tool
    # (``harness.comms.SubagentComms``). Typed ``Any`` for the same import-
    # cycle reason ``jobs`` is a Protocol: ``harness.comms`` imports this
    # module. A CHILD carries its PARENT's instance — that is what lets
    # ``hub`` inside a subagent reach the agent that delegated to it, and
    # what ``is_child(job_id)`` uses to decide which shape of the tool to
    # advertise. ``None`` means no subagent engine, and the tool is then not
    # advertised at all (createIf).
    subagent_comms: Any | None = None
    # The interactive-question hook behind the ``ask`` tool: mount a picker and
    # hand back what the user chose. Installed only by a host that OWNS a
    # terminal it can draw on (the TUI, via
    # ``SessionProtocol.set_ask_handler``), which is what makes its absence the
    # honest capability signal — a subagent inherits ``has_ui`` from its parent
    # but is built without this hook, and a delegated child that advertised
    # ``ask`` would block on a human who is watching the parent's screen and
    # was never shown the question. ``None`` means the tool is not advertised
    # at all (createIf), for the same reason ``wake_scheduler`` is.
    ask_user: AskUserFn | None = None
    # Optional host hook that makes a mid-session credential change VISIBLE to
    # the model (``Session.journal_credential_change``). The ``ask`` tool
    # stores secret answers through ``context.variables`` directly, so the
    # store write succeeds with no session in sight — but without this hook
    # nothing tells the model a key just appeared, and the live failure
    # (session 835fbcafdc27) was the model guessing names for ten minutes.
    # ``None`` (bare tool tests, hosts without a session) degrades to a
    # silent store: the credential still works through bash injection.
    journal_credential: Any | None = None


ToolExecuteFn = Callable[
    [
        str,
        dict[str, Any],
        "AbortSignal | None",
        Callable[[AgentToolUpdate], None] | None,
        ToolContext,
    ],
    Awaitable[ToolResult],
]


#: Renders the human sentence an approval prompt shows for one call. Takes the
#: call's parsed arguments and the session's working directory, and returns
#: ``"<verb>: <target>"`` — the shape the read-tier tools already use —
#: optionally led by one of two hazard markers:
#:
#: * ``[outside workspace]``: the target resolved, and it is not under the root.
#: * ``[unresolvable]``: the target could not be characterised at all, so nothing
#:   can be said about where it is. A describer may also return this in front of
#:   a sentence that is NOT ``<verb>: <target>`` — ``[unresolvable] unparsed url:
#:   <raw>`` — precisely because no verb-and-target pair could be determined.
#:
#: Both markers escalate identically; they differ only in the words the renderer
#: spells out, because a target visibly inside the workspace described as being
#: outside it teaches the reader to distrust the clause that matters. The cwd is
#: a parameter and not read
#: from the process because a session can be rooted anywhere (the server and the
#: scheduler both pass one), and "outside the workspace" is measured against the
#: session's root or it means nothing.
#:
#: This exists because the fallback the loop can build unaided
#: (``name({...json...})``) is the wrong string to put in front of a human who is
#: deciding whether to authorise something: the decision-relevant argument is
#: buried between quoting and irrelevant fields, and no amount of clever
#: truncation in the UI can recover which end of a JSON blob matters. Only the
#: tool knows which of its arguments IS the decision.
ApprovalDescribeFn = Callable[[dict[str, Any], str], str]


class AgentTool(BaseModel):
    """A tool the model can call.

    ``parameters`` is a JSON Schema object (pydantic model's
    ``model_json_schema()`` output). ``concurrency`` controls batch
    scheduling: ``"shared"`` tools run in parallel, ``"exclusive"`` alone.
    ``interruptible`` tools may be aborted mid-run to deliver steering.

    ``describe_approval`` is what the approval prompt says. Every write/exec
    tier tool should set it; without one the loop falls back to a JSON dump,
    which is legible to a reviewer of logs and not to a user answering a
    question under time pressure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    label: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    approval_tier: Literal["read", "write", "exec"] = "exec"
    concurrency: Literal["shared", "exclusive"] = "shared"
    interruptible: bool = False
    hidden: bool = False
    execute: ToolExecuteFn = Field(exclude=True)
    describe_approval: ApprovalDescribeFn | None = Field(default=None, exclude=True)
    #: Per-CALL tier override. A tool whose tier is the highest of its ops
    #: (``hub``: resume starts a session, so the tool is write-tier) still has
    #: read-only ops (``list``, ``peek``) that must not prompt; this hook lets
    #: the call's arguments pick the tier. ``None`` means the static tier.
    call_approval_tier: Callable[[dict[str, Any]], Literal["read", "write", "exec"]] | None = Field(
        default=None, exclude=True
    )


# ---------------------------------------------------------------------------
# Abort signal
# ---------------------------------------------------------------------------


class AbortSignal:
    """asyncio-flavored AbortSignal. Composable via :meth:`any_of`.

    The loop and every tool receive one; aborting sets the event, and long
    operations ``await signal.wait()``-race their work against it.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None
        # Watcher tasks created by ``any_of``; kept so they can be cancelled
        # when the combined signal fires or is no longer needed (otherwise
        # they leak for the lifetime of the watched signals).
        self._watchers: set[asyncio.Task[None]] = set()

    def abort(self, reason: str = "aborted") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()
        self._cancel_watchers()

    def cancel(self) -> None:
        """Cancel any watcher tasks without aborting (the combined signal is
        no longer needed — e.g. the run that wired it has ended)."""
        self._cancel_watchers()

    def _cancel_watchers(self) -> None:
        watchers, self._watchers = self._watchers, set()
        for task in watchers:
            if not task.done():
                task.cancel()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def watchers(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(self._watchers)

    async def wait(self) -> None:
        await self._event.wait()

    @staticmethod
    def any_of(*signals: "AbortSignal") -> "AbortSignal":
        """Combine signals: aborts when any input aborts. Watcher task
        references live on the combined signal and are cancelled when it
        fires or :meth:`cancel` is called. With no running event loop the
        watchers cannot be created — return an already-aborted signal rather
        than silently dropping aborts."""
        combined = AbortSignal()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            combined.abort("no running event loop")
            return combined

        for sig in signals:
            if sig.aborted:
                combined.abort(sig.reason or "aborted")
                return combined

        async def _watch(sig: "AbortSignal") -> None:
            await sig.wait()
            combined.abort(sig.reason or "aborted")

        for sig in signals:
            task = loop.create_task(_watch(sig))
            combined._watchers.add(task)
            task.add_done_callback(combined._watchers.discard)
        return combined


# ---------------------------------------------------------------------------
# Events — the engine→UI boundary
# ---------------------------------------------------------------------------


#: The discriminator literal of a concrete event, defaulting to plain ``str``
#: so an unparameterized ``AgentEvent`` still means "any event". Declaring it
#: covariant is what lets a handler take ``AgentEvent`` and receive any
#: concrete subclass while each subclass keeps its exact ``Literal`` type —
#: the discriminator is written once at construction and never reassigned.
EventTypeT = TypeVar("EventTypeT", bound=str, default=str, covariant=True)


class AgentEvent(BaseModel, Generic[EventTypeT]):
    """Base event. ``type`` discriminates; UIs match exhaustively."""

    model_config = ConfigDict(extra="allow")

    type: EventTypeT


class AgentStartEvent(AgentEvent[Literal["agent_start"]]):
    type: Literal["agent_start"] = "agent_start"
    # Per-session monotonic turn counter; lets UIs drop a superseded
    # agent_end that arrives after the next agent_start.
    generation: int = 0


class AgentEndEvent(AgentEvent[Literal["agent_end"]]):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)
    aborted: bool = False
    error: str | None = None
    generation: int = 0
    # A post-turn compaction happens after the loop creates this event but before
    # the session releases it. Keep the billed messages intact while letting the
    # session replace their now-invalid pre-compaction occupancy reading.
    context_tokens: int | None = None


class TurnStartEvent(AgentEvent[Literal["turn_start"]]):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(AgentEvent[Literal["turn_end"]]):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)


class MessageStartEvent(AgentEvent[Literal["message_start"]]):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(AgentEvent[Literal["message_update"]]):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    delta: str = ""  # incremental text for this update (UIs should append, not re-read)


class MessageEndEvent(AgentEvent[Literal["message_end"]]):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class HistoryDeltaEvent(AgentEvent[Literal["history_delta"]]):
    """Settled transcript rows that became durable while no frontend painted them.

    Emitted by a reconnecting follower for the durable gap between what it
    painted before losing the owner and the fresh sync's cursor. It is a
    HISTORY contract, not a live one: every row is already settled, so the
    consumer must project each row through the same role-aware settled-history
    renderer a cold resume uses — user rows as user rows, assistant prose and
    tool calls as prose and tool cards paired with their results, custom rows
    through their own block paths. Replaying the gap as per-row live
    ``message_end`` events collapsed every role into assistant speech (review
    round 3, MAJOR-1/U7/D1), which is exactly the failure this event exists to
    make unrepresentable.

    ``messages`` preserves durable order and carries tool-result rows beside
    the calls that asked for them, because the settled renderer pairs them by
    ``tool_call_id`` rather than by adjacency.
    """

    type: Literal["history_delta"] = "history_delta"
    messages: list[AgentMessage] = Field(default_factory=list)


class ToolCallComposeEvent(AgentEvent[Literal["tool_call_compose"]]):
    """The model is STILL WRITING a tool call; nothing has run yet.

    Emitted while the arguments stream in, because for a large one they stream
    for a long time: asking for a file of any size means tens of kilobytes of
    `content` arriving token by token, and until the last one lands there is no
    call to execute, no `tool_execution_start`, and — before this event — nothing
    on screen. The user watches a spinner for minutes and reasonably concludes
    the agent has hung. It has not; it is dictating.

    ``argument_bytes`` is the running size of the arguments seen so far, which is
    the only honest progress signal available: the model never says how much is
    left. ``tool_call_id`` is the provider's id when it has arrived and an
    index-derived placeholder before that, so a UI can correlate this with the
    ``tool_execution_start`` that eventually follows.

    ``intent`` is the model's own ``i`` narration, scraped out of the partial
    JSON as soon as its string closes and ``None`` until then — never a
    half-word, because a label growing character by character on a repainting
    row reads worse than no label. This is the highest-value place the field
    appears: ``argument_bytes`` says the agent is alive, and only the intent
    says what for, across the longest silence of the turn.
    """

    type: Literal["tool_call_compose"] = "tool_call_compose"
    tool_call_id: str
    tool_name: str
    argument_bytes: int = 0
    intent: str | None = None


class ToolExecutionStartEvent(AgentEvent[Literal["tool_execution_start"]]):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None


class ToolExecutionUpdateEvent(AgentEvent[Literal["tool_execution_update"]]):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    partial_result: AgentToolUpdate


class ToolExecutionEndEvent(AgentEvent[Literal["tool_execution_end"]]):
    """A tool finished. ``is_error`` mirrors ``result.is_error``.

    The flag is kept as a serialized field because UI clients and the JSON
    exec stream read it directly, but it is NOT an independent input: a
    producer that sets only ``result.is_error`` (or only the flag) would
    otherwise ship an event whose two halves disagree, and a UI reading the
    flag renders a failed tool as a success. The validator ORs them so the
    two can never drift.
    """

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: ToolResult
    duration_s: float | None = None
    is_error: bool = False

    @model_validator(mode="after")
    def _sync_error_flag(self) -> "ToolExecutionEndEvent":
        if self.result.is_error and not self.is_error:
            object.__setattr__(self, "is_error", True)
        return self


class NoticeEvent(AgentEvent[Literal["notice"]]):
    type: Literal["notice"] = "notice"
    text: str
    kind: Literal["info", "warning", "error"] = "info"


class WakeDeliveredEvent(AgentEvent[Literal["wake_delivered"]]):
    """A scheduled wake's prompt was handed to the session for delivery.

    Carries the FULL formatted text so a front end can render an expandable
    receipt (the collapsed line names the wake; the expansion is the message).
    ``catchup`` marks the aggregated resume prompt — several overdue wakes
    folded into one — which renders differently and, being user-attributed,
    must not also replay as a user row.

    Not a NoticeEvent: a wake delivery is expandable content (the delivered
    prompt), not a one-line statement, and a notice has no body to expand.
    """

    type: Literal["wake_delivered"] = "wake_delivered"
    text: str
    catchup: bool = False
    #: Identity of the delivery, so a front end can dedup its live receipt
    #: against the persisted ``wake_prompt`` on a later history replay. Empty
    #: for a catch-up (it folds several wakes and is never replayed).
    wake_id: str = ""
    occurrence: int = 0


class PeerMessageDeliveredEvent(AgentEvent[Literal["peer_message_delivered"]]):
    """A message from ANOTHER local lop session (`lop send`) was delivered here.

    Fires the instant the message lands in this session's transcript/context,
    even while the session is idle, so the owner TUI can paint the
    cross-session indicator immediately rather than waiting for the next turn
    render. Carries ``body`` (the raw text the human reads) and ``sender`` (the
    advisory pid/conversation/model identity for the indicator label).

    Modeled on ``WakeDeliveredEvent`` (which also fires before/around a turn):
    ``message_id`` lets a front end dedup this live receipt against the
    persisted ``peer_message`` row on a later history replay, so a resumed
    session does not double-paint the same delivery.
    """

    type: Literal["peer_message_delivered"] = "peer_message_delivered"
    body: str
    #: Advisory sender identity (pid/session_id/conversation_name/model_label/
    #: cwd). All fields optional — an older/leaner sender still delivers.
    sender: dict[str, Any] = Field(default_factory=dict)
    #: Transcript entry id of the persisted peer message, for replay dedup.
    message_id: str = ""


class SteeringDeliveredEvent(AgentEvent[Literal["steering_delivered"]]):
    """Queued steering messages have entered the model's context.

    Emitted by the session at the moment its steering queue is DRAINED, which is
    the only moment anything can honestly say a mid-turn message was delivered.
    ``steer()`` is fire-and-forget by design — it drops a message on a queue the
    loop empties at its next tool/message boundary — so before this there was no
    signal at all between "queued" and the agent's eventual reply, and a front
    end that told the user "queued" had nothing to correct it with.

    ``count`` is how many messages went in together: a user who sent three lines
    while a tool ran has them all delivered at one boundary, and a receipt per
    message would claim three deliveries where there was one.

    Not a NoticeEvent: this is a state transition a UI RECONCILES against (the
    queued row it already painted), not a line to print. A notice would append a
    second row saying the first row is now wrong.
    """

    type: Literal["steering_delivered"] = "steering_delivered"
    count: int = 1


class SubagentStartEvent(AgentEvent[Literal["subagent_start"]]):
    """A child session was registered as a background job.

    Subagent events are NOT the parent loop's own boundary events: they relay
    one CHILD session's lifecycle onto the parent's stream so a front end can
    render the child's progress. ``job_id`` is the AsyncJob id every event of
    this subagent carries, which is what lets a UI group them.
    """

    type: Literal["subagent_start"] = "subagent_start"
    job_id: str
    label: str
    agent_id: str | None = None


class SubagentProgressEvent(AgentEvent[Literal["subagent_progress"]]):
    """A throttled relay of a child session's activity.

    The relay emits one of these on tool starts/ends and message ends —
    NEVER on every stream delta — because a child streaming a file token by
    token would otherwise flood the parent stream with per-delta events.
    ``progress`` is a short human-readable description of the step.
    """

    type: Literal["subagent_progress"] = "subagent_progress"
    job_id: str
    label: str
    progress: str


class SubagentEndEvent(AgentEvent[Literal["subagent_end"]]):
    """A child session settled. The front end renders completion from THIS
    event; the runner deliberately adds no NoticeEvent or transcript write
    for it, so there is exactly one delivery path."""

    type: Literal["subagent_end"] = "subagent_end"
    job_id: str
    label: str
    status: str  # completed | failed | cancelled
    result_text: str | None = None
    error_text: str | None = None


class CompactionStartEvent(AgentEvent[Literal["compaction_start"]]):
    """A compaction pass began. ``reason`` is ``context-window`` for the
    automatic trigger, ``manual`` when the user asked for it."""

    type: Literal["compaction_start"] = "compaction_start"
    reason: str


class CompactionEndEvent(AgentEvent[Literal["compaction_end"]]):
    """A compaction pass settled.

    ``tokens_before`` is the figure the gate acted on (``max(provider
    context, local estimate)``); ``tokens_after`` is that figure minus the
    history-only saving, measured by one local ruler on both sides with
    archive frames priced at the provider's image billing — so a host can
    report what the pass ACHIEVED in numbers that agree with the status band
    and the next provider bill, and can subtract the pair from its own
    reading without double-counting the request overhead the pass never
    touched. Compaction is slow and its effect is invisible in the
    transcript, so "context compacted" alone asks the user to take it on
    faith. Both are zero when the pass failed, and ``strategy`` is the
    concrete mechanism that ran (``snapcompact`` or ``context-full``).
    """

    type: Literal["compaction_end"] = "compaction_end"
    reason: str
    success: bool
    strategy: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    #: Optional one-clause explanation appended to the receipt, for a pass
    #: whose timing the numbers alone do not explain. The compaction advisor
    #: (BETA) sets it when it pulled the trigger below the configured
    #: threshold: without it an early pass reads as the trigger misfiring.
    #: Optional and defaulting to ``None`` so a host that predates it (and
    #: every ordinary size-triggered pass) is unchanged.
    detail: str | None = None


class RetryStartEvent(AgentEvent[Literal["retry_start"]]):
    type: Literal["retry_start"] = "retry_start"
    attempt: int
    error: str
    fallback_model: str | None = None


class ModelChangeEvent(AgentEvent[Literal["model_change"]]):
    """The model ACTUALLY SERVING requests changed mid-session.

    Emitted when a provider fallback pins a different model (``is_fallback``
    True), when the primary route recovers and requests return to the selected
    model (``is_fallback`` False), and when the user switches models. The
    fallback notice ("provider failure — falling back to …") narrates the
    moment; this event is what lets a front end keep its MODEL DISPLAY truthful
    for the rest of the fallback's lifetime — without it the status band keeps
    asserting the selected model while every request goes elsewhere.

    ``provider``/``model_id`` name the model now in force (the fallback while
    one is pinned, the selected model otherwise); ``effort`` is the reasoning
    level that model is actually running at, which matters because a fallback
    target may clamp the user's chosen level to its own ladder.
    """

    type: Literal["model_change"] = "model_change"
    provider: str
    model_id: str
    effort: str | None = None
    reason: str = ""
    is_fallback: bool = False
    # The model-in-force's own window, carried so consumers that hold only the
    # event (a parent relaying a child's stream) can keep their usage
    # denominators truthful without re-resolving the model themselves. Zero
    # means the emitter did not know, and readers keep their previous value.
    context_window: int = 0


class RetryEndEvent(AgentEvent[Literal["retry_end"]]):
    type: Literal["retry_end"] = "retry_end"
    success: bool


EventHandler = Callable[[AgentEvent], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# Loop configuration — the host extension surface
# ---------------------------------------------------------------------------


class LoopConfig(BaseModel):
    """Everything the loop needs from its host, injected as callbacks.

    The ``AgentLoopConfig`` configuration record. Only ``convert_to_llm``
    and a model streamer are required; everything else has a neutral default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # The model to call (see providers registry). A SNAPSHOT: hosts that can
    # change the model while a run is in flight supply ``get_model`` as well,
    # and this stays the value the run started on.
    model: "ModelSpec"

    # The model to call NOW, re-read immediately before every provider call.
    #
    # A run is not one provider call, it is a chain of them: model, tools,
    # model, tools. ``model`` above is bound once when the host builds this
    # config, so a switch made while a turn is running — the TUI's ``/model``
    # and the picker, which a user reaches precisely BECAUSE the current model
    # is doing badly — could not reach any call in that turn, however many
    # remained. It landed on the session, the status band repainted, and the
    # agent went on calling the old model until the user's next message, which
    # reads as the switch having been ignored.
    #
    # A callback rather than a mutable field because the model lives on the
    # host (``Session._model``) and this config is a per-run value object: the
    # host would otherwise have to hold a reference to the config of whatever
    # run happens to be live and write through it. The loop asks instead, so
    # the session stays the single owner of which model is current.
    #
    # The boundary is deliberately BETWEEN calls, never inside one. An
    # in-flight request keeps the spec it was issued with, so a switch cannot
    # tear down a stream that is already producing tokens or split one response
    # across two models.
    #
    # ``ModelSpec | None`` in the return, and the None is part of the contract
    # rather than defensive slack: it lets a host say "nothing better than the
    # snapshot right now" (still starting, spec briefly unavailable) without
    # having to invent a spec, and the loop treats it exactly like having no
    # resolver at all.
    get_model: Callable[[], "ModelSpec | None"] | None = Field(default=None, exclude=True)

    # The system blocks to send NOW, re-read immediately before every provider
    # call for the supplied model snapshot. A run may contain several
    # model→tools→model steps, and session-scoped
    # instructions such as ``/goal`` live in the volatile tail of these blocks.
    # Keeping only the turn-start snapshot makes a goal changed while a tool is
    # running miss the next model step and wait for another user turn. As with
    # ``get_model``, the in-flight request is never mutated; changes land only at
    # the safe boundary between provider calls. Supplying the model prevents the
    # prompt's model label and the request model from tearing across an async
    # block build. ``None`` keeps the snapshot in
    # ``LoopContext.system_blocks`` for embedders that do not expose live blocks.
    get_system_blocks: Callable[["ModelSpec"], Awaitable[list[str]] | list[str] | None] | None = (
        Field(default=None, exclude=True)
    )

    # Required: render transcript messages (incl. custom entries) into the
    # LLM-visible list sent to the provider.
    convert_to_llm: Callable[[list[AgentMessage]], list[Message]] = Field(exclude=True)

    # Required in practice: stream one assistant response. Providers implement
    # this; the loop only knows the signature.
    stream_fn: Callable[["ChatRequest", AbortSignal | None], Any] = Field(exclude=True)

    # Host context shaping, applied before every provider call.
    transform_context: (
        Callable[[list[AgentMessage]], Awaitable[list[AgentMessage]] | list[AgentMessage]] | None
    ) = Field(default=None, exclude=True)

    # Redact stored session-credential values out of tool RESULTS before the
    # message enters the transcript. A tool can legally echo a secret it was
    # handed (``cat key.pem`` after a bash `echo $TOKEN > key.pem`), and bash
    # alone redacting left ``read``/``grep``/fetch as open exfiltration paths.
    # One hook here covers every tool at the single choke point where a result
    # becomes a message. ``None`` means no credentials are stored.
    redact_tool_result: Callable[[str], str] | None = Field(default=None, exclude=True)

    # Steering (CONSUMING) interrupts tool batches; peek (non-consuming) is
    # polled between calls. Asides never interrupt.
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = Field(
        default=None, exclude=True
    )
    has_steering_messages: Callable[[], bool] | None = Field(default=None, exclude=True)
    get_aside_messages: Callable[[], Awaitable[list[Aside]]] | None = Field(
        default=None, exclude=True
    )
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = Field(
        default=None, exclude=True
    )

    # Gates and hooks.
    before_model_call: Callable[[], Awaitable[bool] | bool] | None = Field(
        default=None, exclude=True
    )
    # Called at the safe boundary after each tool batch lands (before the
    # next model call). May return a replacement ``context.messages`` list —
    # the loop swaps it in and prunes its own run accumulator to the
    # replacement's survivors, so a host that compacts mid-run (automatic
    # mid-turn compaction) never double-persists summarized history. The
    # replacement must keep surviving messages' ids stable: the accumulator
    # filter matches by id.
    on_turn_end: (
        Callable[
            [list[AgentMessage]],
            Awaitable[list[AgentMessage] | None] | list[AgentMessage] | None,
        ]
        | None
    ) = Field(default=None, exclude=True)
    on_before_yield: Callable[[], Awaitable[None] | None] | None = Field(default=None, exclude=True)

    # Fallback routing for unknown tool names (e.g. deferred MCP tools).
    resolve_fallback_tool: Callable[[str], AgentTool | None] | None = Field(
        default=None, exclude=True
    )

    # Urgency counterpart to ``has_steering_messages``: a peek that returns
    # True only when queued steering may cancel a RUNNING tool. The plain
    # peek feeds the immediate-interrupt poll, and courtesy injections (a
    # scheduled wake riding the busy path) share the steering queue with user
    # steers — without this split a wake's timer landing mid-`bash` would
    # kill the tool, exactly the interruption wakes exist not to cause. User
    # steers stay immediate; the session wires this to "a non-wake message is
    # queued". SUBSET, not superset: plain ≥ urgent always holds, so wiring
    # them the other way round would make every courtesy wake an interrupt.
    # ``None`` falls back to the plain peek so existing hosts keep immediate
    # semantics for everything they queue.
    has_urgent_steering_messages: Callable[[], bool] | None = Field(default=None, exclude=True)

    # A fork was requested and is waiting for a safe boundary to clone the
    # transcript at. Polled ALONGSIDE the steering peek (see
    # ``AgentLoop._peek_interrupt``) rather than through a second poll loop,
    # which would double the wakeups on every interruptible tool for no benefit.
    #
    # Deliberately NOT a steering message, which is the shape it superficially
    # resembles: a steer becomes a user turn in THIS session's context and
    # changes what this session is doing, whereas a fork must leave the parent's
    # conversation untouched — the entire point of forking is to try a direction
    # without leaving the one that got you here.
    #
    # The asymmetry that follows from that, and the subtlest rule in the
    # feature: a pending fork may CANCEL an ``interruptible=True`` tool (those
    # are re-runnable by construction, which is what the flag means), so the
    # boundary arrives in ~250 ms instead of after a ten-minute ``wait``. But it
    # must NEVER cause the remaining calls in a batch to be SKIPPED. Steering
    # skips them because the user redirected the work; a fork has redirected
    # nothing, and skipping would silently damage the parent's turn. The
    # batch-skip test in ``_execute_tool_calls`` therefore stays gated on
    # steering alone.
    has_pending_fork: Callable[[], bool] | None = Field(default=None, exclude=True)

    # The provider-reported context size (``Usage.context_tokens``) of this
    # conversation's last call BEFORE this run — the cross-turn seed. The loop
    # reads it for the run's first request and stamps it as
    # ``ChatRequest.context_tokens_hint``; from the second request on, the
    # count the previous call of the SAME run reported wins, because a long
    # tool loop grows past the TTL threshold long before the host's figure is
    # next refreshed. Per-CONVERSATION, stamped per REQUEST by the loop that
    # owns the call — never remembered on the stream fn: subagents share the
    # parent's stream fn (one httpx pool, one failover cascade), so a hint
    # held there is last-writer-wins between the parent and every child, and
    # a child's construction would silently downgrade the parent's next
    # request to 5m on its large context (the exact expiry this hint exists
    # to dodge) while the child's tiny first request inherits the parent's
    # 300k count and pays 2x write rates on a fresh ~10k prefix. The loop asks
    # the HOST (which owns the conversation), exactly as it does for the
    # model and the system blocks. ``None`` (no callback, or nothing reported
    # yet) means no hint and the client's own byte estimate decides.
    get_context_tokens_hint: Callable[[], int | None] | None = Field(default=None, exclude=True)

    interrupt_mode: Literal["immediate", "wait"] = "wait"
    # Epoch-ms deadline for the whole run, if any.
    deadline: float | None = None

    # Guardrails.
    max_paused_turn_continuations: int = 8


# ---------------------------------------------------------------------------
# Provider wire contract
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """A provider/model pair plus the knobs wire clients need."""

    provider: str  # registry id: openai, anthropic, kimi, xai, ollama, ...
    model_id: str
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    supports_tools: bool = True
    supports_images: bool = True
    supports_prompt_cache: bool = False
    # Public OpenAI Responses routing is a model capability, not a provider-wire
    # guess: compatibility providers may serve the same model id while exposing
    # only chat/completions.
    supports_responses_api: bool = False
    base_url: str | None = None  # override for OpenAI-compatible endpoints
    temperature: float = 0.2
    top_p: float = 0.9
    reasoning: bool = False
    # Explicit provider reasoning level. Fallback routes may change providers,
    # so the effort rides on the resolved spec rather than global session state.
    reasoning_effort: str | None = None
    # Whether the model accepts ``temperature``/``top_p`` at all. Some families
    # (Anthropic's Claude 5 generation, OpenAI's reasoning models) reject the
    # parameters outright with HTTP 400, so the defaults above are unsendable
    # and the wire clients must omit the keys rather than send a value.
    # Derived once in ``build_model_spec`` so the wire clients stay free of
    # model-name knowledge; see the note there for why it keys on the model
    # rather than the provider.
    supports_sampling_params: bool = True
    # The reasoning-effort ladder this model accepts, ASCENDING, and the level
    # it is currently set to. Same division of labour as the flag above: derived
    # once in ``build_model_spec`` from ``model.effort`` so no wire client and no
    # widget has to recognise a model name, and the KEY IS OMITTED rather than
    # sent with a null when the model exposes no knob — a provider that rejects
    # the key rejects it just as hard with an empty value.
    #
    # Two fields rather than one because the pair answers two different
    # questions asked by different callers: the wire clients need "what do I
    # send", the status band and ``/effort`` need "what else could this be set
    # to". An empty ``reasoning_efforts`` is the non-reasoning model, and it is
    # what makes ``/effort`` able to say so instead of accepting a level the
    # request would silently drop.
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    # The model's HUMAN name as metadata resolution found it — "Claude Opus 5"
    # for ``anthropic/claude-opus-5``, "MoonshotAI: Kimi K2" for an OpenRouter
    # id no registry row covers. Carried on the spec rather than looked up by
    # the reader because ``build_model_spec`` already holds the resolved
    # ``ModelInfo`` and then threw the name away, which left the status band
    # with nothing to print but the selector and left an aggregator model — the
    # case with no curated name at all — permanently unnamable without a disk
    # read inside a repaint.
    #
    # A RAW name, not a display decision: ``model/naming.py`` owns whether it is
    # safe to show and how it narrows. Empty means resolution had none, which is
    # different from "no name exists" and is why the readers fall back rather
    # than treat this as authoritative.
    display_name: str = ""


class ChatRequest(BaseModel):
    """One provider call. System prompt is a LIST of blocks so providers can
    place cache breakpoints per block (stable instruction block first,
    volatile context last)."""

    model: ModelSpec
    system_blocks: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    tools: list[AgentTool] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] = "auto"
    # A reasoning-effort ceiling the harness set DELIBERATELY (an empty
    # truncation retry stepped the effort down one rung). The session's
    # frozen auto-effort override must not raise the request back above it:
    # the override exists to hold a classification steady, not to undo a
    # retreat the loop made because the higher rung produced nothing.
    effort_ceiling: str | None = None
    # Stable request-prefix identity used by providers' server-side prompt
    # caches. Session hosts populate it once from their session id; keeping it
    # on the request lets retries and fallback clones preserve the same value.
    prompt_cache_key: str | None = None
    #: The provider-reported size of this session's context on its LAST call
    #: (``Usage.context_tokens``), when the host knows it. A HINT, not wire
    #: content: the Anthropic client reads it to decide whether the request is
    #: large enough for the 1-hour prompt-cache TTL (see
    #: ``AnthropicClient._cache_ttl_for``). The client cannot measure this
    #: itself without a tokenizer, and the provider's own count from the
    #: previous turn is the most accurate figure anyone has. ``None`` on a
    #: session's first call and on paths that never saw a usage event (a fork's
    #: first request, one-shot errands); the client then falls back to a byte
    #: estimate of the serialized body. Stamped by the OWNER of the
    #: conversation the call belongs to — the harness loop for turn calls
    #: (``LoopConfig.get_context_tokens_hint`` seeds it, then the run's own
    #: usage events advance it) and the session for its direct calls (asides,
    #: the compaction advisor) — never by the shared stream fn, which merely
    #: passes through what the request carries: subagents share one stream
    #: fn, so any memory held there is last-writer-wins across conversations.
    #: Retries and fallback clones keep it, and an EXPLICIT ``0`` suppresses
    #: the hint (a one-shot prompt that is a fresh write-once prefix).
    context_tokens_hint: int | None = None
    #: This call's output has NOT been shown to anyone yet, so a failed attempt
    #: may be discarded and retried whole.
    #:
    #: Transport policy rather than wire content (no ``_build_body`` reads it):
    #: it lives here because the loop's ``stream_fn`` signature is
    #: ``(request, signal)`` in every host and fake, and the one fact the
    #: failover driver is missing is a property of the CALL.
    #:
    #: ``False`` for a turn and for an aside: their deltas reach the transcript
    #: as they arrive, and a retry would re-render text the user already read.
    #: ``True`` for the one-shot errand that collects the whole stream before
    #: returning a string — the compaction summary — where a stalled read
    #: (``_guarded_chunks`` gives up after 180s of silence) used to be a
    #: permanent failure because the driver had already forwarded events it
    #: could not take back. A failed compaction is not cosmetic: the context it
    #: was meant to shrink keeps growing. Auto-naming is the opposite case and
    #: sets ``isolated`` instead — see below.
    replayable: bool = False
    #: This call is DECORATION running alongside a user turn, and it must not be
    #: able to change anything the turn depends on.
    #:
    #: Transport policy like ``replayable`` above, and it exists because
    #: auto-naming stopped waiting for the turn to finish. A title that arrives
    #: after the work is done is a title nobody needed, so the naming call now
    #: runs CONCURRENTLY with the turn — and a second in-flight request shares
    #: more than bandwidth with it. SIX pieces of session-wide state sit in the
    #: path of an ordinary request, and each is a live route by which a
    #: decorative failure could degrade the user's turn. Each line names where
    #: the denial is enforced, because an enumeration with an unlisted member is
    #: worse than no enumeration:
    #:
    #: 1. ``FailoverRouteState`` is session-sticky. A naming failure that walked
    #:    to a fallback target would ``activate`` it with a 60-second cooldown,
    #:    moving the TURN onto the fallback model — and a naming SUCCESS on the
    #:    primary would ``clear`` a pin the turn is relying on.
    #:    *Denied in* ``stream_with_failover``: ``route_state = None``, which
    #:    kills the target narrowing, the ``activate`` and the ``clear``.
    #: 2. ``AuthStore.rotate_sibling`` mutates the session's sticky credential,
    #:    so an auth failure on a title would re-point the turn's account.
    #:    *Denied in* ``stream_with_failover``: ``retry.enabled = False``, which
    #:    also removes the fallback chain and the backoff budget — every
    #:    rotation path sits behind it.
    #: 3. ``SessionStreamFn`` consumes a pending message boundary to classify
    #:    auto-effort. Whoever arrives first spends it, so a naming call would
    #:    freeze the turn's effort from ITS prompt and emit an "auto effort"
    #:    notice for a request the user never made.
    #:    *Denied in* ``SessionStreamFn.__call__``: the isolated branch returns
    #:    before the classification.
    #: 4. The quota preflight can block a credential and activate a fallback
    #:    route for the whole session.
    #:    *Denied in* ``SessionStreamFn.__call__``: same early return, which is
    #:    also what leaves ``_message_boundary_pending`` unspent (the preflight
    #:    is what clears it).
    #: 5. The session's prompt cache key identifies a request PREFIX. A naming
    #:    call's prefix is a different system block, so sharing the key buys no
    #:    hit and writes a competing entry under the turn's name.
    #:    *Denied in* ``SessionStreamFn.__call__``: same early return, so the
    #:    key is never copied onto the request.
    #: 6. The credential CASCADE mutates routing state on what looks like a
    #:    read: ``AuthStore._resolve`` blocks an OAuth row whose refresh raises
    #:    (so ``_usable_key_rows`` hides it from every later resolve, and the
    #:    turn re-resolves on each tool-loop request) and writes or clears the
    #:    session's sticky credential on the way through its tiers. A transient
    #:    failure on the token endpoint during a title call could therefore
    #:    block the credential the turn is transacting on and repoint stickiness
    #:    to a sibling — the "cold cache prefix, alternating identity headers"
    #:    failure ``create_stream_fn`` warns about. This one is upstream of both
    #:    switches above, so neither reaches it.
    #:    *Denied in* ``_resolve_access_for_provider``, which passes
    #:    ``read_only=request.isolated`` into ``get_oauth_access`` /
    #:    ``get_api_key`` → ``AuthStore._resolve``: no ``block_credential``, no
    #:    ``_set_sticky`` write and none cleared.
    #:
    #: So an isolated request gets exactly ONE attempt on the model it names:
    #: no fallback chain, no sticky route read or written, no credential
    #: rotation, no backoff sleep, no preflight, no boundary classification, no
    #: routing decision taken by its credential resolve, and not the session's
    #: cache key. It still resolves credentials under the session id, so that
    #: READ lands on the same account the turn is on whenever that account is
    #: usable, which is the point. What it cannot do is take the turn anywhere:
    #: if its own resolve finds the sticky credential's refresh broken it may
    #: serve ITSELF from a sibling, but the sticky pointer and the block list
    #: come out of the call exactly as they went in, so the turn's next resolve
    #: still lands where it did before. A successful OAuth refresh does persist
    #: the rotated token, which is that account's own bookkeeping rather than a
    #: decision about where requests go. It fails fast and alone, which is what
    #: lets the caller swallow the failure (see
    #: ``session.naming.generate_title``) without the turn ever knowing a second
    #: call happened.
    #:
    #: Enforced in three places, tested in three: ``stream_with_failover``
    #: (1, 2, and the retry budget), ``SessionStreamFn.__call__`` (3, 4, 5) and
    #: the read-only resolve (6). That the naming call actually SETS this flag
    #: is tested separately, over a real ``Session`` and a capturing stream fn.
    isolated: bool = False


class StreamTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    delta: str


class StreamToolCallDelta(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: str | None = None
    name: str | None = None
    argument_delta: str = ""


class StreamUsageEvent(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage


class StreamEndEvent(BaseModel):
    type: Literal["end"] = "end"
    stop_reason: str  # stop | length | toolUse | refusal | error | aborted
    usage: Usage | None = None
    provider_payload: dict[str, Any] | None = None
    #: The provider's own words about an abnormal end. For ``refusal`` this is
    #: the refusal message (or a line naming the provider's terminal marker when
    #: it sent no prose). Refusals used to be mapped onto ``stop``, which ended
    #: the turn with a clean frame and NOTHING on screen — the user saw an empty
    #: turn and could not tell a refusal from a no-op, let alone decide to
    #: switch models. The wire clients are the only place the provider's actual
    #: marker (``content_filter``, ``refusal``, ``SAFETY``…) is still visible,
    #: so they must put it here; downstream only ever sees the normalized stop.
    error: str | None = None


StreamEvent = StreamTextDelta | StreamToolCallDelta | StreamUsageEvent | StreamEndEvent
