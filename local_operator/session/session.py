"""Session facade — composes loop + tools + transcript + wake + jobs.

The one object every front end talks to (see ``SessionProtocol``). It wires
the host callbacks of :class:`~local_operator.harness.types.LoopConfig` to
session state and keeps the loop itself ignorant of persistence, wakes, and
event fan-out.

Ported semantics:

- Event handlers receive ``AgentEvent``s in emission order; a handler may be
  sync or async, and one raising never breaks the others (isolated with a
  warning).
- Steering messages interrupt tool batches (``interrupt_mode="immediate"``);
  ``steer()`` is the public injection point. Courtesy injections — a wake
  that fires mid-turn — are the exception: they ride the queue but are
  excluded from the interrupt poll (``_has_urgent_steering``), so they join
  at the next successful tool boundary instead of killing the running call.
- Wakes persist as a ``wake_schedules`` custom transcript entry (newest wins)
  and deliver through the prompt path as a user-attributed
  ``wake_prompt`` custom message. A wake delivered into an ongoing turn
  carries resume guidance in its text (``_append_busy_resume_note``): handle
  the wake's task, then continue the work it landed in.
- Compaction is checked after each turn via a LAZY import of
  ``local_operator.compaction.api`` — a missing module degrades to
  no-compaction. Binding wiring: prune tool outputs BEFORE the trigger math,
  trigger on ``compaction_context_tokens`` against the single resolved
  threshold ``min(threshold_percent * window, threshold_tokens)`` (defaults
  80% and 600k, resolved only by ``compaction.thresholds``), strategy
  resolution with snapcompact preferred for vision models, and the recovery
  band gating auto-continuation.
- ``agent_start``/``agent_end`` carry a per-session monotonic ``generation``
  so UIs can drop a superseded ``agent_end`` arriving after the next start.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import logging
import os
import tempfile
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from local_operator.compaction.marker import (
    build_compaction_marker,
    render_compaction_marker,
    replayed_user_message,
)
from local_operator.compaction.tokens import IMAGE_TOKEN_ESTIMATE, approx_text_tokens
from local_operator.harness.approval import ApprovalGate
from local_operator.harness.comms import HUB_MESSAGE_TYPE, SubagentComms
from local_operator.harness.jobs import (
    JOB_RESULT_MESSAGE_TYPE,
    AsyncJob,
    AsyncJobManager,
)
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.subagent import run_subagent
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    Aside,
    AsideResult,
    AskUserFn,
    BrowserSurface,
    ChatRequest,
    CompactionEndEvent,
    CompactionStartEvent,
    Content,
    CustomMessage,
    EventHandler,
    ImageContent,
    LoopConfig,
    Message,
    MessageStartEvent,
    ModelChangeEvent,
    ModelSpec,
    NoticeEvent,
    PeerMessageDeliveredEvent,
    StaleAside,
    SteeringDeliveredEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolContext,
    ToolExecutionEndEvent,
    ToolResult,
    Usage,
    WakeDeliveredEvent,
)
from local_operator.harness.wake import (
    LOAD_GRACE_MS,
    MAX_ARM_MS,
    WAKE_PROMPT_MESSAGE_TYPE,
    WAKE_SCHEDULES_CUSTOM_TYPE,
    DueWake,
    MissedWakeOccurrence,
    WakeSchedule,
    WakeScheduler,
    format_duration,
    format_wake_delivery_text,
)
from local_operator.imaging import rebound_oversize_image
from local_operator.incidents import (
    SESSION_CREDENTIAL_MESSAGE_TYPE,
    SESSION_INCIDENT_MESSAGE_TYPE,
    SESSION_MODEL_SWITCH_MESSAGE_TYPE,
)
from local_operator.session.goal import GoalState
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.naming import (
    CONVERSATION_NAME_CUSTOM_TYPE,
    MAX_TITLE_CHARS,
    ConversationName,
)
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import (
    TODO_REMINDER_MESSAGE_TYPE,
    open_todos,
    restore_todos,
    todo_fingerprint,
    todo_snapshot,
)

if TYPE_CHECKING:
    # Type-only: the session must never pull the MCP stack in at import time.
    # It only holds the manager the composition root hands it.
    from local_operator.mcp.manager import McpManager

    # Type-only for the same reason ``resume`` is imported lazily at its two
    # call sites below: ``cli.py``'s startup path is guarded by a test that
    # fails if resolving ``--resume`` drags the engine in, and that guard runs
    # in the other direction too.
    from local_operator.resume import SessionAttachment

logger = logging.getLogger(__name__)

#: Transcript custom-entry type recording which model is ACTUALLY serving
#: requests when a provider fallback pins a route away from the selected
#: model. Written on every route edge (fallback pinned / primary recovered)
#: and read back at construction, so a resumed session keeps running — and
#: displaying — the model that was really answering when it closed, instead
#: of silently re-routing the first prompt to the provider that was failing.
#: An entry with ``active: None`` records the recovery, which is what lets
#: ``latest_custom``'s backward scan land on "no fallback pinned" after one.
ACTIVE_ROUTE_CUSTOM_TYPE = "active_model_route"

#: Transcript custom-entry type recording the model the user explicitly
#: SELECTED mid-session (``/model <provider>/<id>``), so a resumed session
#: comes back on it instead of silently reverting to the boot default. The
#: sibling of :data:`ACTIVE_ROUTE_CUSTOM_TYPE`, which records where a
#: provider FALLBACK routed requests; this one records where the USER did.
#: Without it a ``/model`` switch survived exactly as long as the process:
#: quit and ``--resume`` replayed the whole conversation onto the config
#: default, which contradicts what the transcript itself shows the user
#: choosing.
#:
#: Each row snapshots ``boot`` — the selector the session was CONSTRUCTED
#: with — beside the selection, so the restore can tell a journalled switch
#: that still applies from one stranded by a changed boot selection (a
#: ``/model default`` write, an edited agent profile, an explicit
#: ``--hosting``/``--model`` flag on the resume itself). A changed boot
#: selection wins: it is the newer, more deliberate choice.
SELECTED_MODEL_CUSTOM_TYPE = "selected_model"

#: Transcript custom-entry type holding a snapshot of the subagents this
#: session launched (see ``SubagentComms.snapshot``) plus their job rows (see
#: ``AsyncJobManager``). Both structures live only in memory, so without this a
#: resumed session opens with an empty subagent panel and no way to reach —
#: let alone resume — the children the previous process started, even though
#: their transcripts survive on disk. Re-snapshotted (newest-wins, like every
#: custom entry) whenever the roster moves; loaded once at construction.
SUBAGENT_ROSTER_CUSTOM_TYPE = "subagent_roster"
SUBAGENT_ROSTER_SIDECAR = "subagent-roster.v1.json"
_SUBAGENT_ROSTER_VERSION = 1
_SUBAGENT_SUMMARY_CHARS = 500

#: Transcript custom-entry type holding the session's todo list. The todo tool
#: keeps the live list in a module-level table keyed by session id (see
#: ``local_operator.tools.builtin.TODO_STORE``), which the process loses on
#: exit; this is the durable copy a resume rehydrates so the todo panel and the
#: continuation guardrail come back exactly where they were. Snapshotted after
#: every turn (the only place the list can have moved) and on demand.
TODO_SNAPSHOT_CUSTOM_TYPE = "todo_snapshot"

#: How many times the title writer will chase a name that moved under its own
#: append before giving up and leaving the rest to the dispose flush. A cap
#: rather than "until it settles" so the bound is structural: the chase is a
#: loop over a value another coroutine can keep changing, and the unbounded
#: form raised ``RecursionError`` after ~3000 passes when driven adversarially.
#: Three is far above what a human renaming a conversation can produce, and the
#: cost of exhausting it is one stale row that the next write corrects.
_NAME_PERSIST_MAX_PASSES = 3

#: Total budget for landing the title at teardown, shared by the wait for an
#: in-flight write and the retry behind it (see
#: :meth:`Session._flush_conversation_name`) so the two cannot add up.
#:
#: Bounded at all because teardown must not hang on a wedged filesystem or
#: behind a transcript lock a long turn still holds. Not TIGHT, because the
#: bound decides whether a slow-but-real write is LOST: an append is
#: sub-millisecond on a local disk, so seconds already mean a stalled mount, and
#: a 4.9 s append lost its title under two 2 s budgets that a single 5 s budget
#: keeps. Five seconds is comfortably above any real append and still a pause a
#: person will sit through once on ctrl+d.
_NAME_FLUSH_TIMEOUT_S = 5.0

#: Self-prompt scheduled after a compaction pass that cleared the recovery
#: band, so the model resumes where the summary left off.
_CONTINUATION_PROMPT = (
    "The conversation context was just compacted into a summary. "
    "Continue the task from where it left off."
)

#: Cap on auto-continuation turns within one user turn
#: MAX_PAUSED_TURN_CONTINUATIONS): a compaction pass that keeps clearing the
#: recovery band must not re-prompt forever.
_MAX_CONTINUATIONS = 8

#: The builtin tools whose createIf gate reads a field only a SESSION can fill
#: (``subagent_launcher``, ``jobs``, ``wake_scheduler``, ``subagent_comms``, the
#: ask hook). Named here rather than inline in
#: :meth:`Session._merge_capability_tools` because two other places have to
#: agree with it: ``harness/subagent.py`` DERIVES a child's prune set from what
#: the merge added, and ``set_ask_handler`` re-runs one entry of it. A tool
#: added to the registry with a session-gated builder and not added here is
#: advertised to nobody.
SESSION_CAPABILITY_TOOLS: tuple[str, ...] = ("task", "wait", "jobs", "wake", "hub", "ask")


class _PeerArrival:
    """Session-owned :class:`PeerArrivalProtocol` behind ``ToolContext``.

    Satisfies the tool-side contract with the smallest possible surface: a
    blocking ``wait`` parks on :meth:`event` and decides with :meth:`count`.

    The event is only ever SET and the count only ever INCREMENTS, which is
    the whole correctness argument. A consumer snapshots the count before
    parking and compares it after waking, so a message that lands between two
    parks is still seen; an ``is_set()`` check plus a producer-side ``clear()``
    would drop exactly that message, and the resulting lost wakeup would delay
    delivery by a whole turn while looking intermittent. Do not "simplify"
    this to a bare Event.

    Three producers share it (see ``PeerArrivalProtocol``): the peer receive
    path, the busy-turn wake delivery, and a child's ``hub`` note. The
    per-kind tally exists only so the woken tool can say which one it was;
    every kind still bumps the single ``count`` the consumer compares, so
    adding a kind never needs a consumer change.
    """

    __slots__ = ("_event", "_count", "_arrivals")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._count = 0
        self._arrivals: dict[str, int] = {}

    def event(self) -> asyncio.Event:
        return self._event

    def count(self) -> int:
        return self._count

    def arrivals(self) -> Mapping[str, int]:
        # A copy: the consumer keeps its snapshot across an await, and the
        # producer keeps incrementing the live dict underneath it.
        return dict(self._arrivals)

    def mark(self, kind: str = PEER_MESSAGE_MESSAGE_TYPE) -> None:
        """Record an arrival of ``kind`` and wake anything parked on it."""

        self._count += 1
        self._arrivals[kind] = self._arrivals.get(kind, 0) + 1
        self._event.set()


@dataclass(frozen=True, slots=True)
class _CompactionPlan:
    """One compaction pass, decided but not yet committed.

    Exists so the automatic gate and the manual ``/compact`` share a single
    decision AND a single commit (:meth:`Session._plan_compaction` /
    :meth:`Session._run_compaction`) instead of two entry points that would be
    free to disagree about the strategy, the cut point or the events.

    ``compaction_api`` is the lazily imported module, carried on the plan
    because the import is the first thing the decision does and the commit
    needs the same one (a missing package degrades to no compaction, so it can
    never be a module-level import).
    """

    compaction_api: Any
    settings: Any
    strategy: str
    #: The CONVERTED, wire-legal history (``_convert_to_llm`` output), not the
    #: transcript vocabulary: the commit slices it into ``to_summarize`` for
    #: ``_produce_summary`` and re-seats ``kept`` behind a fresh
    #: ``CustomMessage`` marker, so ``CustomMessage`` entries are already gone
    #: by the time a plan exists.
    llm_history: list[Message]
    cut: int
    #: Context size for the transcript entry: ``max(provider-reported, local)``.
    context_tokens: int
    #: Local estimate over the pre-pass history — the receipt's "before".
    tokens_before: int
    #: The advisory that CAUSED this pass, or ``None`` when size alone would
    #: have fired it. Carried for the RECEIPT (the user is owed an explanation
    #: when a pass fires below the configured threshold) and for the anti-thrash
    #: bookkeeping in ``_maybe_compact``, both of which mean "the advisor did
    #: this" — so a pass the ordinary trigger would have made anyway leaves this
    #: ``None`` even when a hint was consumed and widened the cut (see the
    #: attribution note where the plan is built). It is not an input to the
    #: commit: ``_run_compaction`` never reads it, so a hint can never reach the
    #: summarizer or permanent history.
    advisor_hint: Any | None = None


@dataclass(frozen=True, slots=True)
class _PendingCompaction:
    """A compaction pass whose SUMMARY is done but which has not been applied.

    Exists because an advisor-triggered pass runs off the critical path: the
    summarization provider call is the expensive half (20-50 s on a large
    context), and awaiting it inline made the user wait mid-conversation for a
    pass the advisor fired EARLY and OFTEN. So the call happens in the
    background against a SNAPSHOT and the cheap half — rebuilding the context —
    is applied later, at a safe boundary.

    That split is only sound because the transcript keeps moving while the call
    is in flight, and the result must not be applied to a conversation it no
    longer describes. ``summarized_ids`` is what makes that checkable: the ids
    of the prefix this summary actually covers, in order. At apply time the
    live render's first ``plan.cut`` entries must still be exactly those ids.

    Why that single check is sufficient, rather than a broader diff: the commit
    keeps ``render[plan.cut:]``, so history APPENDED while the pass ran is
    inside the kept window automatically and is never at risk. The only way to
    lose history is for the PREFIX to have changed — a competing pass, a
    rebuild — and an unchanged prefix proves that did not happen. A pass that
    fails the check is discarded, never repaired: a stale pass that silently
    drops new history is the worst failure this feature can have, and
    discarding costs only the summary call.
    """

    plan: _CompactionPlan
    summary: str
    preserve_data: dict[str, Any] | None
    reason: str
    #: Ids of ``plan.llm_history[: plan.cut]`` — the span the summary replaces.
    summarized_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ForkRequest:
    """A ``/fork`` waiting for a safe boundary in a RUNNING turn.

    Frozen: the request is decided once, by the host that took the user's
    command, and the boundary only executes it. Nothing about which
    conversation is being branched, or with what opening message, may change
    between the request and the clone.
    """

    #: Where the sessions store lives. Carried on the request rather than read
    #: at drain time so the clone cannot land in a different store than the one
    #: the command was issued against.
    config_dir: Path
    #: The optional first user message for the fork (``""`` for none).
    message: str
    #: Called with the new session id once the clone lands, or with ``""`` when
    #: it failed. The host owns everything user-visible from there (the receipt,
    #: the spawn, the notice weight); the session's job ends at the clone.
    on_complete: Callable[[str, str], None]


def _configured_max_running(values: Mapping[str, Any] | None = None) -> dict[str, int]:
    """``{"max_running": N}`` from config, or ``{}`` to keep the default.

    The concurrent-job ceiling governs how many subagents (and backgrounded
    bash jobs — they share one pool) may run at once. The right number is a
    property of the operator's machine and models rather than of this code, so
    it is configurable under ``values.subagents.max_running``, beside the
    ``values.subagents.models`` tiers the launcher already reads.

    Returns kwargs rather than a value so an unset or unusable config is
    expressed by passing NOTHING, leaving ``AsyncJobManager``'s own default as
    the single source of truth for it. Duplicating the default here is how the
    two would later disagree.

    ``values`` is the ``values`` mapping to read from. Passed by the config
    watcher's listener (``_apply_config_change``) so the live re-read applies
    exactly the validation the build applied; ``None`` reads the file through
    a fresh ``ConfigManager`` as the constructor always has.

    Never raises: a malformed config must not stop a session from starting.
    A non-positive value is rejected rather than honoured, because 0 would
    deadlock every launch behind a gate that can never open.
    """
    try:
        if values is None:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            raw = ConfigManager(config_dir()).get_config_value("subagents", None)
        else:
            raw = values.get("subagents")
        if not isinstance(raw, Mapping):
            return {}
        value = raw.get("max_running")
        if value is None:
            return {}
        parsed = int(value)
        if parsed < 1:
            logger.warning(
                "subagents.max_running=%r must be >= 1; using the built-in default", value
            )
            return {}
        return {"max_running": parsed}
    except Exception:  # noqa: BLE001 — a bad config must not fail session startup
        logger.warning("subagents.max_running could not be read; using the built-in default")
        return {}


def _coerce_compaction_settings(settings: Any) -> Any:
    """Defensive coercion (CL-01 belt): a dict-shaped ``compaction_settings``
    is validated into ``CompactionSettings``; already-typed or ``None`` pass
    through. The factory coerces too — this keeps direct constructors (the
    server facade, benchmarks, tests feeding a raw config dict) safe."""
    if settings is None or not isinstance(settings, dict):
        return settings
    try:
        from local_operator.compaction.api import CompactionSettings
    except ImportError:
        return None
    try:
        return CompactionSettings.model_validate(settings)
    except Exception as exc:  # noqa: BLE001 — degrade, never break the session
        logger.warning("Invalid compaction settings dict, using defaults: %s", exc)
        return CompactionSettings()


def _archive_to_json(archive: Any) -> dict[str, Any]:
    """JSON-safe dump of a snapcompact ``Archive``.

    Thin on purpose: ``Archive`` owns BOTH directions of the base64 contract
    now (a ``frames`` validator that decodes, a json serializer that encodes).
    This used to encode by hand here while revival went through pydantic's lax
    ``str``->``bytes`` coercion, which UTF-8-encodes base64 text rather than
    decoding it — the halves drifted and every post-compaction request shipped
    doubly-encoded PNGs. Kept as the single named call site so the reason the
    dump must be ``mode="json"`` (raw PNG bytes are not UTF-8) stays written
    down where the callers can see it.
    """
    return archive.model_dump(mode="json")


def _callable_accepts_one_positional(func: Callable[..., Any]) -> tuple[bool, bool]:
    """``(accepts_one_positional, certain)`` for ``func``.

    Used to decide whether the system-blocks provider takes the live
    ``model_label`` (the factory and subagent providers do; a legacy zero-arg
    callable does not). When the signature can be read the answer is certain;
    when it cannot (a builtin, a C callable, some mocks) the answer is ``True``
    but ``certain`` is ``False``, so the caller tries the labelled call and may
    fall back once. Distinguishing the two matters: a TypeError raised INSIDE a
    provider whose signature was read as one-arg is a real bug that must surface,
    not be swallowed by a zero-arg retry.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True, False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            return True, True
    return False, True


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default transcript→LLM rendering.

    ``compaction_summary`` markers become a user message carrying the summary;
    a snapcompact archive in ``preserve_data`` is rendered back into
    text_head → imaged middle → text_tail blocks (base64 ``ImageContent``
    between ``TextContent`` edges). ``fork_boundary`` and ``wake_prompt``
    deliveries become user messages of their formatted text, and the newest
    ``todo_reminder`` (only the newest) becomes one too; other custom entries
    are dropped (bookkeeping never enters LLM context). ``provider_payload``
    rides along untouched.
    """
    out: list[Message] = []
    # Only the NEWEST todo reminder survives the render. An earlier one asserts
    # a todo list that has since changed, so replaying it would hand the model a
    # stale — and by then actively false — claim about its own state, and
    # re-argue a nudge it has already answered. The pruning belongs here because
    # the renderer is a pure function of the whole list and reminders are never
    # persisted, so nothing downstream could do it. Older ones simply fall
    # through to the allow-list's drop.
    newest_reminder = -1
    for index in range(len(messages) - 1, -1, -1):
        if _is_todo_reminder(messages[index]):
            newest_reminder = index
            break
    for index, message in enumerate(messages):
        if isinstance(message, Message):
            out.append(message)
        elif message.custom_type == "compaction_summary":
            # Pass the ORIGINAL entry id through the render: the transcript
            # persists custom entries with their CustomMessage.id, so a
            # compaction cut landing on a rendered marker can still locate
            # ``first_kept_entry_id`` on replay.
            out.append(_render_compaction_marker(message, entry_id=message.id))
        elif message.custom_type in (
            SESSION_INCIDENT_MESSAGE_TYPE,
            SESSION_MODEL_SWITCH_MESSAGE_TYPE,
            SESSION_CREDENTIAL_MESSAGE_TYPE,
        ):
            # An incident rides the sender's preformatted text (the classifier
            # already wrote category + suggested action), exactly like a wake
            # delivery: it must reach the model as a user turn or the session
            # stays blind to why its last run died. A model-switch record uses
            # the same path so the model becomes aware it is now answering as a
            # different model (a deliberate switch or a failover fallback),
            # rather than only seeing a changed static "Model:" system line.
            # A credential record rides the same path so a mid-session
            # ``/credential`` is ANNOUNCED to the model rather than only
            # changing the prompt tail, which the model has no reason to
            # re-read (the failure behind session 835fbcafdc27).
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
        elif message.custom_type in (
            "fork_boundary",
            WAKE_PROMPT_MESSAGE_TYPE,
            HUB_MESSAGE_TYPE,
            JOB_RESULT_MESSAGE_TYPE,
            PEER_MESSAGE_MESSAGE_TYPE,
        ):
            # A hub message renders exactly like a wake delivery: the sender
            # already formatted ``details["text"]``, and it must reach the
            # model as a user turn or the agent it was addressed to never
            # sees it. A peer message (`lop send` from another local session)
            # rides the same path: it MUST be listed here or the human sees the
            # cross-session transcript row but the model never does. Unlisted
            # custom types are dropped (bookkeeping), which is precisely the
            # trap a new aside type falls into.
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
        elif message.custom_type == TODO_REMINDER_MESSAGE_TYPE and index == newest_reminder:
            # The continuation guardrail's nudge (``Session._todo_continuation``)
            # reaches the model as a user turn or it does nothing at all: this
            # allow-list is the trap a new aside type falls into, and a dropped
            # reminder would make the loop re-enter with nothing to react to.
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
    return out


def _todo_reminder_text(pending: list[dict[str, str]]) -> str:
    """The nudge the continuation guardrail injects (``_todo_continuation``).

    Wrapped in ``<system-reminder>`` and labelled as harness-injected: the model
    reads it as a user turn, and without the label it would answer the user
    about a message the user never sent. Items are listed verbatim so the model
    resolves the texts the tool will actually match, and the three honest exits
    are named because "keep going" alone is what produces work marked done that
    was never done. Compact on purpose — it is injected at every boundary.
    """
    items = "\n".join(f"- {item['text']}" for item in pending)
    return (
        "<system-reminder>\n"
        "Injected by the harness at the turn boundary. Not from the user, and "
        "not shown to them.\n"
        f"These todo items are still open:\n{items}\n"
        "Do not end the turn while items are open. Either keep working, or "
        "resolve each one honestly: `todo done` when it is finished, "
        "`todo block` with a reason when it needs a user decision or an "
        "external service, `todo drop` when it is no longer needed. If the "
        "decision is the user's to make, put it to them with the `ask` tool.\n"
        "</system-reminder>"
    )


def _is_todo_reminder(message: AgentMessage) -> TypeGuard[CustomMessage]:
    """Is ``message`` a live continuation nudge (``_todo_continuation``)?

    One predicate for the three places that have to agree about it — the
    renderer's newest-only rule, the expiry scan
    (:meth:`Session._live_todo_reminders`) and the compaction render
    (:meth:`Session._render_for_compaction`). The ``isinstance`` half is
    load-bearing rather than defensive: a RENDERED reminder is a plain
    ``Message`` carrying the same text, and a predicate that matched that too
    would read compaction's own output back as a fresh nudge.
    """
    return isinstance(message, CustomMessage) and message.custom_type == TODO_REMINDER_MESSAGE_TYPE


#: ``CustomMessage`` types that belong in the transcript as message entries.
#:
#: An ALLOW-LIST, because the cost of the two mistakes is asymmetric. Omitting a
#: type that should persist loses one replayed message; persisting one that
#: should not corrupts the session's history for good, and the compaction
#: summary marker is the proof: it is minted by ``_run_compaction`` and stored
#: through ``append_compaction`` as its OWN entry type, so writing it a second
#: time as a message replays a superseded summary back into context beside the
#: live one. A deny-list enumerating the ephemeral types cannot see a type that
#: does not exist yet; this one excludes it by default.
_PERSISTABLE_CUSTOM_TYPES: frozenset[str] = frozenset(
    {
        SESSION_INCIDENT_MESSAGE_TYPE,
        SESSION_MODEL_SWITCH_MESSAGE_TYPE,
        # SESSION_CREDENTIAL_MESSAGE_TYPE is deliberately absent: a credential
        # announcement asserts a LIVE capability ("$KEY is injected into every
        # bash command") against a store that is process-memory-only. A
        # replayed record would tell a restarted session an env var is present
        # when bash will not have it — stale by definition, the same class as a
        # transient model-switch fallback (review round 1, R2). Resume-time
        # discovery is served by the ``<session-credentials>`` prompt-tail
        # block, which is rebuilt from the live store every turn and so
        # correctly shows nothing after a restart.
    }
)


def _pair_spliced_tool_results(messages: list[Message]) -> list[Message]:
    """``messages`` with any message spliced INTO a tool batch moved after it.

    The last line of defence for the invariant Layer 1 (``_append_or_park_journal``)
    enforces at the source: the rendered history must never carry a non-tool
    message between an assistant's ``tool_calls`` and the ``tool_result``s
    answering them. A session already bricked IN MEMORY predates that guard and
    cannot be helped by it, and a future writer appending straight to
    ``_context.messages`` would reintroduce the same 400. This runs at request
    assembly so neither case reaches a provider.

    The constraint is POSITIONAL, not set membership. Verified against the live
    Anthropic API (claude-sonnet-4-5, 2026-08-25), because
    ``AnthropicClient._build_body`` coalesces tool results onto a preceding user
    message and the difference decides what a repair has to do::

        [tool_result, tool_result]            -> 200
        [tool_result, tool_result, text]      -> 200
        [text, tool_result, tool_result]      -> 400
        interloper as its own message between -> 400

    So results must be the LEADING run of the message that follows the calls,
    and the repair is to MOVE the interloper after them. Synthesizing
    placeholder results instead is wrong and was tested as such: it draws a
    different 400 (``unexpected tool_use_id found in tool_result blocks``)
    because the genuine results still arrive behind the placeholders.
    Synthesis is only correct for a genuinely UNANSWERED tail, which is
    :meth:`Session._wire_legal_snapshot`'s job — an unanswered batch is left
    untouched here.

    Relative order among several interlopers is preserved, and moving them is
    safe ONLY because everything that can land in that window is a
    harness-authored notice (``session_model_switch``, ``session_incident``):
    reordering advisory chrome against a tool batch changes nothing the user
    wrote. **If a custom type is ever added that carries user-authored text,
    this assumption needs revisiting** — moving a user's words past a tool
    batch would silently reorder the conversation they see.

    Linear in ``len(messages)``, and that has to stay true because this sits on
    every provider call. Each batch's inner scan stops at the next assistant
    carrying ``tool_calls``, so the scanned windows are disjoint and no message
    is examined by more than one of them. An earlier version let an UNANSWERED
    batch rescan to the end of the list, which made a history of open batches
    quadratic (3.7s at 4000 messages against 2.3ms answered); see
    ``test_pair_spliced_tool_results_scans_each_message_once`` for the speed
    property and
    ``test_pair_spliced_tool_results_does_not_pull_the_next_batch_into_an_open_one``
    for the correctness property, which are pinned separately because the bound
    carries both.
    """
    # Hot path: ``_render_history`` runs on every provider call, and the
    # overwhelming majority of histories have no tool batch to violate.
    if not any(message.role == "assistant" and message.tool_calls for message in messages):
        return messages

    out: list[Message] = []
    index = 0
    total = len(messages)
    repaired = False
    while index < total:
        message = messages[index]
        out.append(message)
        index += 1
        if message.role != "assistant" or not message.tool_calls:
            continue
        # Collect the batch's answers and anything spliced among them, stopping
        # at the first message that is neither — that is the end of the batch,
        # and a trailing non-tool message there is already legal.
        expected = {call.id for call in message.tool_calls}
        results: list[Message] = []
        interlopers: list[Message] = []
        scan = index
        while scan < total and expected:
            candidate = messages[scan]
            if candidate.role == "assistant" and candidate.tool_calls:
                # A new batch opens here, so this one is over whatever is still
                # unanswered. Bounding the scan is what keeps the whole pass
                # linear: every batch's window ends at the next batch's start,
                # so the windows are disjoint and each message is examined by at
                # most one of them. Without this bound an unanswered batch
                # rescanned to the end of the list for EVERY later assistant
                # message -- 4000 messages took 3.7s instead of 2.3ms.
                #
                # It is also the more correct stop, in one specific shape: a
                # batch whose answer arrives LATE, after the next batch has
                # already opened. Only then does ``expected`` empty at all, so
                # only then can the repair fire on a window that spans two
                # batches -- ``A1(x) A2(y,z) T(y) T(z) T(x)`` would come out as
                # ``A1 T(y) T(z) T(x) A2``, with A2 emitted BEHIND its own
                # answers: one illegal shape traded for another.
                #
                # A batch that is never answered at all is NOT this case, and
                # was never at risk: ``expected`` stays non-empty, so the
                # ``if expected: continue`` guard below suppresses the repair
                # with or without this bound. Do not reach for an unanswered
                # batch to justify the stop -- reach for the late answer.
                break
            if candidate.role == "tool":
                results.append(candidate)
                # Guarded rather than ``or ""``: coercing a missing id to the
                # empty string would let an unidentified result answer a call
                # only if some call also had an empty id. A result with no id
                # answers nothing, so it closes no call.
                if candidate.tool_call_id is not None:
                    expected.discard(candidate.tool_call_id)
            else:
                interlopers.append(candidate)
            scan += 1
        if expected or not interlopers:
            # Unanswered tail (leave it to ``_wire_legal_snapshot``) or nothing
            # spliced. Either way this batch is not ours to touch.
            continue
        out.extend(results)
        out.extend(interlopers)
        index = scan
        repaired = True

    if repaired:
        # After Layer 1 this is dead code. Logging at warning rather than debug
        # is deliberate: silence here would hide a regression that reintroduces
        # the splice, and the repair would mask it right up until some future
        # shape it cannot fix.
        logger.warning(
            "repaired a message spliced into an open tool batch; a journal "
            "append bypassed the turn-boundary guard (see _append_or_park_journal)"
        )
    return out


def _paired_prefix(messages: Sequence[AgentMessage]) -> list[AgentMessage]:
    """``messages`` truncated so it never ENDS in unanswered tool calls.

    The durability flushes persist the live context, and that list is not
    always legal to replay. ``AgentLoop`` appends the assistant message the
    moment the model turn ends and appends the tool results only once
    ``_execute_tool_calls`` returns, so for the whole duration of every tool
    batch — the longest part of a turn, and exactly when a Ctrl+C or a crash
    lands — the list ends in an assistant message whose ``tool_calls`` have no
    answers. Persisting that verbatim writes a dangling ``tool_use`` into the
    transcript PERMANENTLY, and the next resume replays it into a 400 on both
    wires ("must be followed by tool messages responding to each
    tool_call_id"). Measured: a Ctrl+C mid-batch left two unpaired calls on
    disk and an unusable session.

    :meth:`Session._wire_legal_snapshot` faces the same illegal tail for a
    request it is about to send and pairs the calls with placeholders. This is
    the persistence counterpart, and it DROPS rather than pairs: a placeholder
    is the honest answer to "what are you doing right now", but on disk it
    would be a permanent lie about a tool that never reported. The unanswered
    assistant message is re-sent by the model on the next run anyway, so
    dropping it loses nothing a resume needs — while everything completed
    before it is kept, which is the whole point of the flush.

    Only the TAIL is trimmed, and "tail" means *up to the last real answer*.
    A ``CustomMessage`` in the tail does NOT prove the list is legal: it is not
    a ``Message``, and one can land on the live context while a tool batch is
    still in flight. ``journal_incident`` appends straight to
    ``_context.messages``, and ``_on_mcp_incident`` fires it through
    ``_spawn_background`` — so an MCP breaker tripping mid-batch leaves
    ``[..., assistant(tool_calls), session_incident]``. A scan that stopped at
    the first non-assistant entry would see the incident, declare the tail
    clean, and persist the unanswered assistant beneath it — the very row this
    function exists to refuse (review round 2, R5; reproduced as
    ``DANGLING: ['c2']``).

    So customs are stepped OVER and kept, while unanswered assistant messages
    beneath them are dropped; the scan stops at the first ``role="tool"``,
    which is a real answer and therefore a genuinely legal tail. Re-listing a
    custom that was already persisted is harmless — ``_persist_new_messages``
    dedups by id.
    """
    out = list(messages)
    keep_tail: list[AgentMessage] = []
    while out:
        tail = out[-1]
        if isinstance(tail, Message):
            if tail.role == "assistant" and tail.tool_calls:
                out.pop()  # unanswered: never persist it
                continue
            break  # a tool result or a plain message: the tail is legal
        # A non-Message (custom) proves nothing about legality. Hold it aside
        # and keep looking underneath it.
        keep_tail.append(out.pop())
    out.extend(reversed(keep_tail))
    return out


def _is_persistable_message(message: AgentMessage) -> bool:
    """Whether ``message`` may be written to the transcript as a message entry.

    Plain ``Message``s always may. A ``CustomMessage`` may only when its type is
    in :data:`_PERSISTABLE_CUSTOM_TYPES` — todo reminders are ephemeral by
    design (a stored reminder replays as a user message the user never sent and
    goes on asserting that finished items are open), and a compaction summary
    marker is already persisted as a compaction entry.

    This matters because the mid-turn gate flushes from the LIVE loop context,
    which is exactly where both of those live: after a pass the context is
    ``[marker, *kept]``, so the very next boundary would otherwise persist the
    marker.

    A TRANSIENT model-switch record (a per-request failover fallback) is a
    special case: it belongs in the live context so the running model knows it
    is on a fallback, but it must NOT be persisted. A transient fallback that
    outlives the process is stale by definition — a resumed session boots on its
    selected model, so replaying "you are now on fallback B (temporary)" with no
    matching recovery would contradict the authoritative ``Model:`` line
    (review R1).
    """
    if isinstance(message, CustomMessage):
        if message.custom_type == SESSION_MODEL_SWITCH_MESSAGE_TYPE and bool(
            message.details.get("transient")
        ):
            return False
        return message.custom_type in _PERSISTABLE_CUSTOM_TYPES
    return True


def _stamped_todo_fingerprint(details: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """The todo fingerprint a reminder was built from, normalized for compare.

    Read back through an explicit tuple/str coercion rather than trusted as
    stored: ``details`` is a plain dict, and any JSON round trip turns the
    nested tuples into lists — a raw ``!=`` would then be True for an unchanged
    list and expire every reminder on sight. A reminder with no stamp (or a
    malformed one) compares equal to nothing and expires, which is the safe
    direction: a nudge that may be lying is worth less than one turn without it.

    ARITY MUST MATCH ``builtin.todo_fingerprint`` (design §5.3): it now emits
    3-tuples ``(phase_name, text, status)``, so this filter keeps ``len == 3``.
    If this stayed at 2 while the source grew to 3, EVERY stamped item would be
    dropped, the stamped side would compare empty against a non-empty current
    fingerprint, and every reminder would expire on every render — the latch
    errs safe (keeps nudging) so 'does it nudge' still passes while the
    no-second-nudge suppression silently breaks. The phased no-second-nudge
    guardrail test is what catches a regression here.
    """
    stamped = details.get("fingerprint") or ()
    return tuple(
        (str(item[0]), str(item[1]), str(item[2]))
        for item in stamped
        if isinstance(item, (list, tuple)) and len(item) == 3
    )


# Relocated to ``compaction.marker`` so hosts that must not import the session
# (the evaluation runner's ``run_compaction_pass``) render a marker exactly as
# this session does. The private names stay bound here because they are the
# session's seam and existing tests reach them through this module.
_replayed_user_message = replayed_user_message


#: Stands in for an image the provider refused, so the turn that follows it
#: still makes sense. A silently shortened message would leave the model
#: reading a summary whose "the screenshots below" no longer has any below.
IMAGE_DROPPED_NOTICE = "[image omitted: the provider rejected it and it has been dropped]"

#: Stands in for an image the ACTIVE model cannot receive, so a turn that
#: follows a switch to a text-only model still makes sense. Distinct from
#: :data:`IMAGE_DROPPED_NOTICE` on purpose: the provider refusal is a sticky
#: session condition the user cannot undo, while this one lasts only as long
#: as the current model does — switching back to a vision model restores the
#: images, and the notice must not claim they are gone for good.
IMAGE_OMITTED_TEXT_ONLY_NOTICE = "[image omitted: the current model does not accept images]"


def _rebound_history_images(messages: list[Message]) -> list[Message]:
    """Shrink any image block in the rendered history that is over the cap.

    The composer and the ``read`` tool both bound images on the way in, so on a
    current build nothing reaching here is oversized and this walk is a cheap
    header check that changes nothing. It exists for the blocks those bounds
    cannot reach:

    - History written by an OLDER build, which is on disk and is replayed
      verbatim on every resume. This is the observed failure: a 2206x266 paste
      from an unbounded build kept earning ``...max allowed size for many-image
      requests: 2000 pixels`` on every single prompt, on a build whose composer
      could no longer create such a block. Bounding on the way in fixed the
      cause and could not fix the sessions already poisoned by it.
    - Snapcompact archive frames, which are rendered to a per-provider geometry
      rather than through :func:`~local_operator.imaging.bound_image_for_model`
      and reach 2048px wide under the Google shape. ``set_model`` can swap
      providers mid-session, so a Gemini-shaped archive can replay to Anthropic
      and breach a ceiling it was never measured against.

    Applied to the RENDERED history and never to the transcript, exactly like
    :func:`_without_images`: the stored frames keep their original resolution,
    so ``/export`` is unaffected and a provider with laxer limits still gets the
    full-size image on a later session.

    This is a REPAIR, not a replacement for the ``is_image_rejection`` degrade.
    It fixes the one cause that is mechanically knowable in advance (the block
    is too big); the degrade stays underneath for every refusal that is not —
    corrupt bytes, a format the provider dislikes, or the same bytes being
    accepted for hours and then refused.
    """
    out: list[Message] = []
    for message in messages:
        if not any(isinstance(block, ImageContent) for block in message.content):
            out.append(message)
            continue
        content: list[Content] = []
        changed = False
        for block in message.content:
            if isinstance(block, ImageContent):
                rebound = rebound_oversize_image(block.data)
                if rebound is not None:
                    data, mime_type = rebound
                    block = block.model_copy(update={"data": data, "mime_type": mime_type})
                    changed = True
            content.append(block)
        out.append(message.model_copy(update={"content": content}) if changed else message)
    return out


def _without_images(messages: list[Message], *, model_incapable: bool = False) -> list[Message]:
    """Every message with its image blocks replaced by a one-line notice.

    Used after a provider has refused an image (see
    :func:`~local_operator.providers.failover.is_image_rejection`), and when
    the active model is KNOWN not to accept images (``model_incapable=True``:
    the session was switched onto a text-only spec while the history still
    carries image blocks). Applied to the RENDERED history rather than to the
    transcript, so nothing is destroyed: the archive keeps its frames,
    ``/export`` still has them, and a later session on a provider that accepts
    them is unaffected.

    The notice wording differs by cause (see
    :data:`IMAGE_OMITTED_TEXT_ONLY_NOTICE`): a provider refusal is permanent
    for the session, a model's incapacity is not, and a notice that apologised
    as if the images were gone for good would lie to the model the moment a
    vision model is switched back in.

    Consecutive images collapse to ONE notice. A snapcompact archive replays as
    fifty-odd frames between two text edges, and fifty identical apology lines
    would cost more context than the summary they are standing in for.
    """
    notice = IMAGE_OMITTED_TEXT_ONLY_NOTICE if model_incapable else IMAGE_DROPPED_NOTICE
    out: list[Message] = []
    for message in messages:
        if not any(isinstance(block, ImageContent) for block in message.content):
            out.append(message)
            continue
        content: list[Content] = []
        for block in message.content:
            if isinstance(block, ImageContent):
                if content and getattr(content[-1], "text", None) == notice:
                    continue
                content.append(TextContent(text=notice))
            else:
                content.append(block)
        out.append(message.model_copy(update={"content": content}))
    return out


def _pruned_ids(messages: Sequence[AgentMessage]) -> set[str]:
    """Ids of messages already carrying the pruning pass's ``pruned`` marker.

    Taken BEFORE a prune pass so the pass's effect is a set difference. The
    pass reports only a boolean ``changed``, and re-journalling every
    already-blanked message on every turn would grow the transcript with
    duplicate entries faster than the blanking shrinks it.
    """
    return {
        message.id
        for message in messages
        if isinstance(message, Message) and (message.provider_payload or {}).get("pruned")
    }


def _should_compact(
    compaction_api: Any,
    context_tokens: int,
    window_tokens: int,
    settings: Any,
    advisory_ok: bool,
) -> bool:
    """``should_compact``, calling the pre-advisor signature when it can.

    The kwarg is passed ONLY when an accepted advisory is actually in play,
    which is both a correctness property and a compatibility one:

    - Correctness: with the beta off, ``advisory_ok`` is always ``False`` and
      the call is byte-identical to the one this code made before the advisor
      existed. There is no "new argument that happens to be inert" for a
      reader to reason about.
    - Compatibility: hosts and tests substitute their own ``should_compact``
      (the suite has several three-parameter doubles), and an unconditional
      kwarg turns every one of them into a ``TypeError`` at the gate. A double
      that predates the advisor cannot be asked about advice anyway.

    The ``TypeError`` fallback covers a double that IS given advice but does
    not know the parameter: degrading to the ordinary trigger is the safe
    direction, since the advisor may only ever make a pass fire earlier.
    """
    if not advisory_ok:
        return bool(compaction_api.should_compact(context_tokens, window_tokens, settings))
    try:
        return bool(
            compaction_api.should_compact(context_tokens, window_tokens, settings, advisory_ok=True)
        )
    except TypeError:
        return bool(compaction_api.should_compact(context_tokens, window_tokens, settings))


#: Minimum fraction of the pre-pass context an ADVISORY compaction has to free
#: before the advisor is judged to be earning its calls. Below it, the advisor
#: is disabled for the session (see :meth:`Session._settle_advisor`).
#:
#: Deliberately a REDUCTION fraction rather than an absolute residual band. The
#: kill switch answers "is the advice reclaiming anything?", which is a
#: property of the pass, not of where the context happens to sit; the previous
#: absolute form disabled the advisor after a measured 39.3% reduction (agent
#: review round 2, major-3).
#:
#: 0.10 is set well below the reductions an advisory pass actually achieves
#: (39-40% across every size measured on the real path) and well above zero,
#: so it catches the failure it exists for — a pass that fires, spends a
#: summary call, and frees essentially nothing — without firing on passes that
#: are working. It is NOT ``RECOVERY_BAND``: that constant answers a different
#: question (is there room to continue the turn?) against a different
#: yardstick (the trigger threshold), and reusing it here is exactly how one
#: residual came to be both "created headroom" and "reclaimed nothing".
_ADVISOR_MIN_RECLAIM_FRACTION = 0.10

#: Task-shaped term of the preserve-window cap: how many ``keep_recent_tokens``
#: the window may grow to before the capacity ceiling is also applied
#: (:meth:`Session._advisor_floor_cap` carries the full derivation, the limits
#: of the evidence for this exact value, and why the capacity term must stay).
#: Sized against 17 measured active-task spans, which are bimodal: thirteen
#: under 54k and four between 113k and 132k. 5x the 20,000 default lands this
#: term at 100,000 — above every ordinary task, below the outlier cluster that
#: drove 35-41% retention. A MULTIPLE rather than an absolute, so it tracks a
#: user who configures a wider verbatim window. It BINDS only above the ~250k
#: window crossover; below that ``threshold // 2`` is the smaller term.
_TASK_FLOOR_KEEP_MULTIPLE = 5


def _advisor_detail(hint: Any | None) -> str | None:
    """Receipt sentence for an advisor-triggered pass, or ``None``.

    An advisory pass fires below the threshold the user configured, so the
    plain "context compacted" receipt would read as the trigger misbehaving.
    Naming the advisor and its own stated reason is what makes an early pass
    legible; ``reason`` is already length-capped at validation
    (``ADVISOR_MAX_REASON_CHARS``), so nothing unbounded reaches the notice.
    """
    if hint is None:
        return None
    reason = getattr(hint, "reason", "")
    if reason:
        return f"advisor: {reason}"
    return "advisor: task boundary"


#: The ONLY ``AsyncJob`` fields the roster snapshot carries. An allowlist, not
#: an exclude set, because the roster projection must stay small: it is written
#: to the replaced sidecar on every roster move, and a superseded copy in a
#: legacy transcript is only reclaimed lazily (``compact_file`` now collapses
#: superseded ``subagent_roster`` customs, but not until the next compaction) —
#: so anything unbounded here bloats both surfaces. The fields kept are all
#: small and bounded: the identity, the timings
#: the panel prices elapsed from, the model/usage/window it paints, and the
#: routing/queue flags a restore needs. Everything a reader might want beyond
#: this — the child's full ``result_text``, its verbatim ``prompt`` (documented
#: "Unbounded on purpose"), the ``trajectory``, the live ``output_tail`` — is
#: recoverable on demand from the child's OWN transcript via ``hub op='peek'``,
#: the same argument the trajectory exclusion already made, carried to its
#: conclusion.
_ROSTER_ROW_FIELDS = frozenset(
    {
        "id",
        "type",
        "status",
        "start_time",
        "started_at",
        "settled_at",
        "label",
        "queued",
        "agent_id",
        "owner_id",
        "model_label",
        "context_window",
        "usage",
        # Bounded by distinct provider/model/accounting-mode tuples, not child
        # count. This is the durable half of nested accounting after a child
        # manager is disposed and therefore must survive process resume.
        "descendant_usage",
        # Folded attempts no longer have visible rows. Their bounded accounting
        # components must travel with the winner so the sidecar can reconstruct
        # the full logical child after a process restart.
        "prior_attempt_usage",
        "restored",
        # Small bounded strings, stamped at registration: a restored row must
        # still say what kind of child it was ("task"/"scout") and at what
        # effort tier, or the resumed panel/page/band would blank both facts on
        # every child a previous process launched — the exact regression the
        # rest of this allowlist exists to prevent for the model/usage fields.
        "agent_role",
        "effort",
        # Resume reconciliation metadata is bounded by attempts of this one
        # logical child and keeps legacy handles valid after process restore.
        "logical_id",
        "attempt_aliases",
    }
)

#: Cap on the one free-text field the row keeps. ``error_text`` drives the
#: panel's one-line summary for a restored FAILED row, so it is worth keeping —
#: but a stack trace or a giant provider error must not reintroduce the
#: unbounded-growth M1 fix removes, so it is clipped to a sentence's worth.
_ROSTER_ERROR_CAP = 2_000


def _subagent_job_row(job: AsyncJob) -> dict[str, Any]:
    """One task job as a slim, bounded dict for the roster snapshot.

    Projects the job onto :data:`_ROSTER_ROW_FIELDS` (see its note for why an
    allowlist rather than an exclude set), plus a length-capped ``error_text``
    so a restored failed row can still say why it failed. ``model_dump(
    mode="json")`` makes the nested ``Usage`` JSON-native.
    """
    row = job.model_dump(mode="json", include=set(_ROSTER_ROW_FIELDS))
    if job.error_text:
        row["error_text"] = job.error_text[:_ROSTER_ERROR_CAP]
    return row


def _compact_subagent_record(record: dict[str, Any]) -> dict[str, Any]:
    """Bound list-surface text while preserving the exact resume directory."""
    compact = dict(record)
    for key in ("prompt", "error_text", "result_text"):
        value = compact.get(key)
        if value is not None:
            compact[key] = str(value)[:_SUBAGENT_SUMMARY_CHARS]
    return compact


def _write_roster_sidecar(path: Any, payload: dict[str, Any]) -> None:
    """Replace current roster state durably without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = path.with_name(os.path.basename(raw_temp))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _write_roster_sidecar_if_changed(
    path: Any, payload: dict[str, Any], previous_fingerprint: str | None
) -> tuple[str, bool]:
    """Compute the roster fingerprint and write the sidecar only if it moved.

    Runs ENTIRELY on the worker thread (dispatched via ``asyncio.to_thread``):
    the fingerprint is an O(roster) ``json.dumps`` and #308's invariant is that
    everything that scales leaves the event loop. Computing it on the loop made
    a large roster pay that serialization on every roster event — the exact
    regression ``TestMeasurementCosts`` catches. Returns ``(fingerprint, wrote)``.

    The fingerprint deliberately EXCLUDES ``generation``: the counter is bumped
    on every roster event by design (it drives the coalescing loop), so
    including it would make every payload unique and defeat the guard. Only the
    durable projection a resume actually reads is compared. ``sort_keys`` keeps
    the serialization order-stable so an unchanged roster hashes identically
    across writes.
    """
    fingerprint = json.dumps(
        {key: payload[key] for key in ("version", "jobs", "records")},
        sort_keys=True,
        separators=(",", ":"),
    )
    if fingerprint == previous_fingerprint:
        return fingerprint, False
    _write_roster_sidecar(path, payload)
    return fingerprint, True


def _read_roster_sidecar(path: Any) -> dict[str, Any] | None:
    """Read a supported sidecar, falling back to the legacy transcript on error."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != _SUBAGENT_ROSTER_VERSION:
            return None
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            return None
        if not isinstance(payload.get("jobs"), list) or not isinstance(
            payload.get("records"), list
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def _parsed_usage(payload: dict[str, Any]) -> Usage | None:
    """One persisted ``usage`` payload as a :class:`Usage`, or ``None``.

    A transcript row is data from a previous process and may predate a field, so
    a payload that no longer validates is dropped rather than raised: a status
    readout must not be able to stop a session from opening. ``None`` simply
    falls through to the next-newest reading, and then to the local estimate.
    """
    try:
        return Usage.model_validate(payload)
    except Exception:
        logger.debug("dropping unparseable persisted usage payload", exc_info=True)
        return None


def _last_reported_usage(usages: Sequence[Usage | None]) -> Usage | None:
    """The newest provider-reported :class:`Usage` in ``usages``, or ``None``.

    Scans BACKWARDS and stops at the first hit: the newest reading is the only
    one that describes the context as it now stands, and a resumed conversation
    can hold hundreds of entries to walk past.

    **Refuses any reading recorded before the newest compaction**, and that
    exception is the whole reason this is a function rather than a one-line
    scan. A compacted transcript replays as a summary marker followed by the
    KEPT WINDOW, and those kept messages still carry the ``usage`` they were
    given BEFORE the pass — figures describing a context that no longer exists,
    which nothing supersedes when the session compacted and then exited.

    Seeding from one is not a small error. Measured on a transcript that
    compacted at 900k of a 1M window, the reading came back 900_000 against a
    real 1_707 — 527x over, installed as EXACT so the correct local estimate
    could never replace it, and handed to ``should_compact``, which would then
    rewrite the user's history on the first turn after the resume.
    Under-reporting was the bug this seeding fixed; this is the same lie
    pointing the other way, and the compaction consequence makes it the more
    expensive of the two.

    The rule cannot be expressed on the replayed list alone. The marker sits at
    the HEAD of it and the kept window FOLLOWS it, so "stop scanning backwards
    at the marker" reads exactly backwards — the stale messages come first — and
    "any marker disqualifies everything" throws away the legitimate case: a
    session that compacted and then ran ten more turns has a perfectly good
    newest reading, and refusing it would send every such resume back to the
    local estimate for no reason.

    So the boundary is taken from the TRANSCRIPT, whose entries are in append
    order and therefore say which readings were recorded after the pass.
    ``entries_after_compaction`` returns exactly those; a history with no
    compaction returns all of them, which is the ordinary path.

    ``None`` means "no usable reading here", a real state and distinct from
    zero: a brand-new session, a conversation of nothing but user messages, a
    provider that reports no usage, or a compacted history with no completed
    turn since the pass. Callers must not collapse the two — a confident 0 on a
    resumed session is the empty-context lie this exists to prevent — and
    falling through to the local estimate is the right answer for all of them.
    """
    for usage in reversed(usages):
        if usage is not None:
            return usage
    return None


_render_compaction_marker = render_compaction_marker


class Session:
    """The session facade. Satisfies ``SessionProtocol``."""

    def __init__(
        self,
        model: ModelSpec,
        stream_fn: Callable[[ChatRequest, AbortSignal | None], AsyncIterator[StreamEvent]],
        tools: Sequence[AgentTool],
        transcript: Transcript,
        *,
        session_id: str | None = None,
        agent_id: str = "main",
        convert_to_llm: Callable[[list[AgentMessage]], list[Message]] | None = None,
        compaction_settings: Any | None = None,
        yolo: bool = False,
        has_ui: bool = False,
        cwd: str | None = None,
        skill_resolver: Callable[[str], str | None] | None = None,
        request_approval: "ApprovalGate | None" = None,
        goal_state: GoalState | None = None,
        #: The variables surface behind list_variables/read_variable. Held by
        #: the SESSION because ``_build_tool_context`` is rebuilt every turn:
        #: a store passed only to the factory's context reached the createIf
        #: check that decides the tools exist, and never the context they run
        #: against, so both tools read a bare process-env store in every
        #: session while a configured store sat unused beside them.
        variables: Any | None = None,
        #: Which background job this session IS, when it is a subagent run.
        #: Carried so a host can tell a delegated call's approval request apart
        #: from the parent turn's — a denial latched during the parent's turn
        #: must not silently kill the tools of a child still running after it.
        #: Same reason ``variables`` is held here: ``_build_tool_context`` is
        #: rebuilt every turn, so anything set only on the construction-time
        #: context reaches the createIf check and never the executor.
        job_id: str | None = None,
        #: The short name this session was DELEGATED under, set only on a
        #: subagent (``zoom-scroll-fix``, ``bridge-qa``). A child never
        #: generates a conversation title — naming runs in the TUI host and the
        #: owned-session runtime, neither of which a one-shot child passes
        #: through — so this plus ``parent_display_name`` below is the
        #: whole of its display identity. Reaches tools via
        #: ``_build_tool_context``; display-only, never an authorization input.
        job_label: str = "",
        #: The parent's own :meth:`_display_session_name`, on a subagent only.
        #: A RESOLVER rather than the parent's title holder, for two reasons
        #: that are really one: it re-reads on every call, so a parent named or
        #: renamed AFTER this child launched still reaches the child's display
        #: surfaces (the common case — the naming errand lands early in the
        #: parent's first turn while children are launched later); and it
        #: resolves TRANSITIVELY, so a grandchild under a middle child whose
        #: own holder is permanently empty still reads the top-level
        #: conversation's title instead of falling through to the shared cwd.
        #: A holder cannot do the second: the middle child has no title of its
        #: own to hold. ``None`` on every top-level session.
        parent_display_name: Callable[[], str] | None = None,
        #: The parent↔child messaging surface (``harness.comms.SubagentComms``).
        #: A top-level session mints its own; a CHILD is handed its parent's,
        #: which is what makes ``hub`` inside a subagent talk to the agent that
        #: delegated to it. Held here for the ``_build_tool_context`` reason
        #: above: a rebuilt context must keep pointing at the same instance.
        subagent_comms: Any | None = None,
        #: The user's persistent agent registry, when the host keeps one. Two
        #: readers: the ``agent`` tool (list/author role profiles) and role
        #: resolution for ``task(agent=...)``. Held on the session for the
        #: ``_build_tool_context`` reason above, and typed ``Any`` to keep the
        #: heavy ``local_operator.agents`` module out of this import graph.
        #: ``None`` means no registry: the ``agent`` tool is not advertised and
        #: delegation falls back to the packaged starter profiles.
        agent_registry: Any | None = None,
        #: The user's persistent team registry, when the host keeps one. Behind
        #: the ``team`` tool and behind ``/team``. ``None`` means the ``team``
        #: tool is not advertised.
        team_registry: Any | None = None,
        conversation_name: ConversationName | None = None,
        # Called each turn with the session's live ``model_label`` so the env
        # block names the running model; accepts it positionally (``...``) and
        # may return sync or async. A provider that ignores the argument is
        # still valid — the label is additive.
        system_blocks_provider: Callable[..., list[str]] | Callable[..., Awaitable[list[str]]],
    ) -> None:
        self._model = model
        self._stream_fn = stream_fn
        notice_bridge = getattr(self._stream_fn, "set_notice_handler", None)
        if callable(notice_bridge):
            # The stream owns provider routing; the session owns ordered event
            # delivery. Binding the two here lets every front end receive quota
            # and fallback notices without teaching the harness loop providers.
            notice_bridge(self._stream_notice)
        route_bridge = getattr(self._stream_fn, "set_route_handler", None)
        if callable(route_bridge):
            # Same division of labour as the notice bridge, for the ROUTE
            # itself rather than its narration: the stream reports which model
            # is actually serving requests, the session persists that fact and
            # emits the event a front end repaints its model display from.
            route_bridge(self._on_route_settled)
        # Deliberately NO bridge for the prompt-cache TTL hint, unlike the two
        # above: the stream fn is SHARED with subagents, so a registered reader
        # is last-writer-wins — constructing a child overwrote the parent's
        # reader for good and downgraded every later parent request to 5m
        # (review F8). The hint rides each request instead, stamped by its
        # owner: the loop reads ``get_context_tokens_hint`` for turn calls and
        # the session's direct calls stamp ``_context_tokens_hint`` themselves.
        self._tools = list(tools)
        self._transcript = transcript
        self._session_id = session_id or transcript.directory.name
        self._agent_id = agent_id
        # The goal rides the prompt's volatile tail; the holder is shared with
        # the system-blocks provider so an edit applies from the next turn.
        self._goal_state = goal_state if goal_state is not None else GoalState()
        self._variables = variables
        self._job_id = job_id
        self._job_label = job_label
        self._parent_display_name = parent_display_name
        self._subagent_comms = subagent_comms
        self.agent_registry = agent_registry
        self.team_registry = team_registry
        #: The team this session is running as manager of, when ``/team``
        #: launched it. Held here so every ``task`` child inherits the group's
        #: collaboration and project briefs without the manager restating them.
        #: ``None`` on an ordinary session.
        self.active_team: Any | None = None
        #: User-facing copy explaining why a stored attachment did NOT come
        #: back ("" when it did, or when there was none). Set by
        #: :meth:`_restore_attachment` for a team or profile that failed to
        #: resolve, whether because it is gone or because the registry was
        #: unavailable.
        #:
        #: NOT named ``…_error``: it holds display copy, and its only consumer
        #: renders it as a ``"warning"`` system notice, deliberately not an
        #: ``"error"`` — a missing team is a recoverable state of this session,
        #: not a failure of it. The name says so, so the next reader does not
        #: route it to an error surface or log it as a fault (D5).
        #:
        #: Held instead of raised because a missing team must never make a
        #: conversation unopenable, and held instead of logged because the
        #: reader who needs it is the user staring at a band segment that
        #: stayed blank.
        self.attachment_restore_notice: str = ""
        #: True only while :meth:`_restore_attachment` is re-applying the stored
        #: attachment, which suppresses the journal write in
        #: :meth:`_persist_attachment`. Without it the restore would write back
        #: through the very mutators it calls, and a PARTIAL restore would erase
        #: the half it could not resolve: a session whose team is momentarily
        #: unresolvable (registry not wired on this host, a team dir not yet
        #: synced) would persist ``team=""`` and lose the name for good, turning
        #: a recoverable miss into permanent data loss. A restore is a READ of
        #: state that is already on disk; it has nothing new to record.
        self._restoring_attachment = False
        #: Stored names the restore could NOT resolve, carried so the next
        #: journal write preserves them instead of erasing them (R1).
        #:
        #: Suppressing the write during the restore is not enough on its own,
        #: and the gap was a real data-loss path: once the restore returns,
        #: ``active_team`` is ``None`` for the half that missed, so the very
        #: next ordinary mutation — a plain ``/goal`` is enough — journals that
        #: emptiness over the surviving name and the attachment is gone for
        #: good, one command after a failure that was supposed to be
        #: recoverable. Keeping the name here makes the recovery survive
        #: arbitrary further use of the session, which is what "transient"
        #: has to mean.
        #:
        #: Cleared per slot by an explicit user action (a successful attach
        #: replaces it, a detach drops it), never by the miss itself — see
        #: :meth:`_persist_attachment`.
        self._unresolved_team: str = ""
        self._unresolved_agent: str = ""
        # The conversation's title. A holder rather than a plain string for
        # the same reason the goal is one: the title arrives on a DETACHED
        # naming task after the host already built its status chrome, and
        # both sides must see the same object rather than a stale copy.
        self._conversation_name = (
            conversation_name if conversation_name is not None else ConversationName()
        )
        #: True while a stored title has not reached the transcript yet. The
        #: write is a background task, so this is what lets dispose tell "the
        #: name is on disk" from "the write was cancelled on the way there".
        self._conversation_name_dirty = False
        #: The in-flight journal write for the title, tracked apart from
        #: ``_background_tasks`` because dispose CANCELS those and this one has
        #: to be awaited instead — see :meth:`_spawn_conversation_name_write`.
        self._conversation_name_task: "asyncio.Future[None] | None" = None
        #: True while this session is a FORK still wearing its parent's title —
        #: the fact behind :attr:`wears_inherited_title`. Defaults False so an
        #: ordinary session (and any host that never reaches the resume path)
        #: is never tagged; ``_load_conversation_name`` sets it from the
        #: ``_is_unnamed_fork()`` verdict it already computes at construction.
        self._wears_inherited_title = False
        #: The ``provider/model_id`` this session was CONSTRUCTED with — the
        #: boot selection resolved from agent > CLI flag > config. Captured
        #: before any restore or switch moves ``_model``, because it is the
        #: reference every ``selected_model`` journal row carries: a resume
        #: whose boot selection no longer matches a row's ``boot`` must NOT
        #: adopt that row (the changed default/flag/profile is the newer
        #: choice). See :data:`SELECTED_MODEL_CUSTOM_TYPE`.
        self._boot_selector = f"{model.provider}/{model.model_id}"
        #: True while a mid-session model selection has not reached the
        #: transcript yet; the same dispose-flush contract as the title (the
        #: write is a background task, and dispose cancels background tasks,
        #: so a switch made just before ctrl+c needs the flush to land it).
        self._selected_model_dirty = False
        #: The in-flight journal write for the selection, tracked apart from
        #: ``_background_tasks`` for the same reason the title's write is:
        #: dispose CANCELS background tasks, and a cancel that lands before
        #: the wrapped coroutine's first await leaves it un-awaited (asyncio
        #: then reports it at GC time) as well as unwritten. The flush AWAITS
        #: this task instead — see :meth:`_spawn_selected_model_write`.
        self._selected_model_task: "asyncio.Future[None] | None" = None
        self._system_blocks_provider = system_blocks_provider
        # Whether the block provider accepts the live ``model_label`` argument.
        # Computed once here rather than per call: the factory and subagent
        # providers do, a bare zero-arg callable (some hosts, most tests) does
        # not, and the label is additive either way. A provider whose signature
        # cannot be inspected (a C callable, a mock) is assumed to take it and
        # falls back at the call site only if that assumption is wrong.
        (
            self._blocks_provider_takes_label,
            self._blocks_provider_arity_certain,
        ) = _callable_accepts_one_positional(system_blocks_provider)
        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        #: Set once a provider refuses a request because of an image block, and
        #: never cleared: from then on this session renders its history without
        #: images. See :func:`~local_operator.providers.failover.is_image_rejection`
        #: for why recovery has to be sticky rather than per-request — the
        #: offending block is IN the history, so an un-degraded retry sends it
        #: again, and the session is otherwise unusable for good.
        self._images_rejected = False
        #: Latch for the text-only-model omission announcement (see
        #: :meth:`_render_history`). The omission itself is not sticky — it
        #: reads the CURRENT spec every render — but its announcement is:
        #: every render of an image-bearing history on a text-only model would
        #: otherwise re-post the notice, and a transcript that repeats the same
        #: warning on every turn is noise the user learns to ignore.
        self._text_only_omission_announced = False
        self._compaction_settings = _coerce_compaction_settings(compaction_settings)
        self._yolo = yolo
        self._has_ui = has_ui
        self._cwd = cwd or "."
        self._skill_resolver = skill_resolver
        self._request_approval = request_approval
        # No constructor kwarg, unlike ``request_approval``: there is no
        # default ask host to fall back to. Only a front end that owns the
        # terminal can draw a picker, so the hook arrives from
        # ``set_ask_handler`` after that front end has its session, and until
        # it does the ``ask`` tool is simply not advertised.
        self._ask_user: AskUserFn | None = None

        self._loop = AgentLoop()
        replayed_messages = list(transcript.build_llm_history())
        # A fork's inherited bytes stay untouched for prompt-cache continuity;
        # its lineage warning exists only at the live context tail and is
        # consumed once, so resume never accumulates synthetic transcript rows.
        from local_operator.fork import consume_fork_boundary

        fork_boundary = consume_fork_boundary(transcript.directory)
        if fork_boundary:
            replayed_messages.append(
                CustomMessage(
                    custom_type="fork_boundary",
                    attribution="system",
                    details={"text": fork_boundary},
                )
            )
        self._context = LoopContext(
            system_blocks=[],
            messages=replayed_messages,
            tools=self._tools,
        )
        self._handlers: list[EventHandler] = []
        # A ``/fork`` requested while a turn is running, waiting for a safe
        # boundary. Not a steering message and not on the steering queue: a
        # steer becomes a user turn in THIS conversation, while a fork must
        # leave it exactly as it was. See :meth:`request_fork`.
        self._fork_pending: _ForkRequest | None = None
        self._steering_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        # Producer identity is transport provenance, not message identity. Keep
        # it beside queued messages so ordinary callers can still choose ids
        # without accidentally entering the mobile admission namespace.
        self._steering_producers: dict[int, str] = {}
        # Hosts reserve producer identities before a steer reaches its durable
        # boundary. A failed append must hand that identity back explicitly;
        # otherwise one disk error consumes both the retry ID and a bounded
        # steering slot for the lifetime of the owner.
        self._steering_rejection_handlers: list[Callable[[str, str], None]] = []
        # Count of courtesy wake_prompt messages sitting in the steering
        # queue. The immediate-interrupt poll may cancel a RUNNING tool only
        # for steering the user actually typed (see ``_has_urgent_steering``):
        # a scheduled wake's timer landing mid-`bash` must ride the next
        # boundary instead of killing work it has no right to interrupt. A
        # count, not message ids: ``queue._queue`` is asyncio-private, and the
        # only producer/consumer ordering question a count cannot answer is
        # settled by the decrement happening at the same drain the delivery
        # does.
        self._courtesy_wake_count = 0
        # Peer-arrival signal behind ToolContext.peer_arrival, so a blocking
        # `wait` can park on "a message reached my mailbox" alongside its job
        # settle events. Owned by the session because the ToolContext is
        # rebuilt every turn and this has to outlive one.
        self._peer_arrival = _PeerArrival()
        # Host-registered teardown (see add_dispose_hook): resources the
        # composition root owns but the session's lifetime governs.
        self._dispose_hooks: list[Callable[[], Awaitable[None] | None]] = []
        # Set by the composition root when MCP servers are wired in, and read
        # only for diagnostics — the session never drives the manager itself,
        # it just governs its lifetime through a dispose hook.
        self.mcp_manager: McpManager | None = None
        # What that wiring actually achieved, recorded as data so a front end
        # can render it. ``None`` means MCP wiring never ran on this session at
        # all, which is distinct from "it ran and found nothing configured" —
        # the latter is an empty outcome. See session/mcp_status.py for why the
        # record does not live in the mcp package.
        self.mcp_startup: McpStartupOutcome | None = None
        # Front-end sink fired once the MCP startup round SETTLES — i.e. every
        # server deferred past the 250 ms gate has reached a terminal state and
        # ``mcp_startup`` has been rebuilt with the combined tally. A full-screen
        # TUI installs it to re-raise the boot toast/notice with the final
        # numbers instead of the provisional gate snapshot; headless callers
        # leave it None (they already print on settle in the factory). Signature:
        # (McpStartupOutcome). See session_factory.wire_mcp_into_session.
        self._on_mcp_startup_settled: Callable[[Any], None] | None = None
        self._aside_thunks: list[Aside] = []
        self._continuation_queue: list[AgentMessage] = []
        # Latest provider-reported usage. SEEDED from the replayed transcript
        # rather than starting at None, because on a resumed session the last
        # turn's usage is a fact that already happened and the transcript is
        # where it was persisted. Two things read it and both were wrong without
        # this: the compaction trigger fell back to a local estimate for the
        # first turn after every resume (the estimate runs 7-17% off, and the
        # gate it feeds decides whether to rewrite the user's history), and
        # ``restored_usage`` below reports the conversation's real size to a
        # front end that would otherwise open on an empty-looking context.
        # From the TRANSCRIPT, not from the replayed context: only append order
        # distinguishes a reading taken after the newest compaction from one the
        # pass invalidated, and the replayed list deliberately loses that (see
        # ``Transcript.usages_since_compaction`` and ``_last_reported_usage``).
        self._last_usage: Usage | None = _last_reported_usage(
            [_parsed_usage(payload) for payload in transcript.usages_since_compaction()]
        )
        # The per-conversation prompt-cache TTL hint (see
        # ``ChatRequest.context_tokens_hint``): the provider-reported context
        # size of THIS session's last turn call, excluding isolated errands
        # (their tiny prefix is not this conversation) and one-shots that do
        # not carry the conversation (their prompt is a different, write-once
        # prefix — see ``_one_shot_complete``). Moves in lockstep with
        # ``_last_usage`` (see ``_note_usage``) so a hint is never staler than
        # the compaction trigger's figure, and is stamped per request by the
        # call's owner: the loop seeds its run from it and then prefers its
        # own in-run counts; asides and the advisor stamp it directly. Seeded
        # from the transcript's last usage so a RESUMED large session starts on
        # its real size instead of re-deriving it through the client's byte
        # estimate. ``None`` until the first provider call reports one.
        self._context_tokens_hint: int | None = (
            self._last_usage.context_tokens if self._last_usage is not None else None
        )
        self._last_activity_ms: int = 0  # epoch ms; drives idle-flush pruning
        self._generation = 0  # monotonic turn counter for agent_start/end
        # Boundary-event suppression across a post-compaction continuation:
        # `_held_end` parks the loop's agent_end until compaction has decided
        # whether the run continues, `_logical_generation` remembers which
        # agent_start the eventual end belongs to. Both are None outside a run.
        self._held_end: AgentEndEvent | None = None
        # The loop's held end owns billing, but a later post-turn compaction owns
        # occupancy. Carry that newer level to the boundary without rewriting the
        # usage objects that lifetime cost and analytics still need.
        self._held_context_tokens: int | None = None
        self._abort_requested = False  # sticky across the continuation gap
        # --- Speculative compaction advisor (BETA; inert unless the config
        # opts in via values.compaction.advisor_enabled). All of it is session
        # state rather than plan state because the advisor runs OFF the turn:
        # the call is spawned at a tool-loop boundary and nothing awaits it, so
        # the result has to be parked somewhere the next plan gate can read it.
        #
        # ``_advisor_hint`` holds at most one validated hint. It is
        # single-slot on purpose: a queue of hints is a queue of stale
        # opinions about a conversation that has since moved, and the plan gate
        # only ever wants the newest.
        self._advisor_hint: Any | None = None
        self._advisor_in_flight = False  # at most ONE call outstanding; skip, never queue
        self._advisor_calls = 0  # against settings.advisor_max_calls
        self._advisor_last_turn = -(10**9)  # turn index of the last call, for every_n_turns
        self._advisor_cooldown_until = -1  # turn index the cooldown expires at
        # Kill switch. An advisor-triggered pass that fails to clear the
        # recovery band proved the advice was not merely early but wrong about
        # there being headroom to reclaim, and repeating it is a compaction
        # treadmill on the user's bill. Non-negotiable: once set, the advisor
        # is done for the life of the session.
        self._advisor_disabled = False
        # --- Asynchronous compaction pass (advisor-triggered path only) ------
        #
        # The advisor call was already off the turn, but the PASS it authorised
        # was awaited inline at every gate, and that pass makes its own
        # summarization provider call. While a pass only ever fired at the
        # ceiling that was invisible: the turn had to stop there anyway. The
        # advisor's whole purpose is to fire EARLIER and more often, which
        # turns the same inline await into a visible stall in the middle of
        # otherwise healthy work.
        #
        # So an advisor-triggered pass runs detached and is applied at the next
        # safe boundary. ``_pending_compaction`` is single-slot for the reason
        # ``_advisor_hint`` is: a queue of passes is a queue of plans about a
        # conversation that has since moved.
        self._pending_compaction: _PendingCompaction | None = None
        # At most ONE detached pass outstanding; skip, never queue. Mirrors
        # ``_advisor_in_flight``, and is what stops a boundary that fires every
        # tool batch from spawning a pass per batch while the first still runs.
        self._compaction_pass_in_flight = False
        # The Task wrapping that pass, so a ceiling pass can CANCEL it rather
        # than run a second summarization alongside it (issue #413). Identity,
        # not a generation: there is at most one, and the latch already
        # enforces that. ``None`` whenever nothing is in flight.
        self._compaction_pass_task: asyncio.Task[Any] | None = None
        # Last completed provider request (epoch ms). Distinct from
        # _last_activity_ms: the idle-flush pruning must measure provider-cache
        # age, and stamping turn bookkeeping right before the check made the
        # 90-minute flush dead code.
        self._last_provider_request_ms = 0
        self._seeded = False  # seed_history is once-only, pre-prompt
        self._logical_generation: int | None = None
        self._fallback_tool_resolver: Callable[[str], AgentTool | None] | None = None
        # Todo-continuation latch: the full todo fingerprint captured at the
        # last guardrail nudge in THIS user turn, so a model that yields twice
        # with a byte-identical list is not nudged a second time. Reset per user
        # turn in _run_turn_pipeline; see :meth:`_todo_continuation`.
        # Both fingerprints are the phase-aware 3-tuple form
        # ``(phase_name, text, status)`` that ``builtin.todo_fingerprint`` now
        # emits (phased-todos change): the reminder latch and the persistence
        # latch both compare against that source, so their annotations must match
        # its arity or a comparison silently degrades.
        self._todo_reminder_fingerprint: tuple[tuple[str, str, str], ...] | None = None
        # The todo fingerprint that was last WRITTEN to the transcript, so the
        # post-turn snapshot only appends when the list actually moved. Seeded
        # from disk on a resume (``_load_todo_snapshot`` sets it) so the first
        # turn after a restore does not re-persist an unchanged restored list.
        self._persisted_todo_fingerprint: tuple[tuple[str, str, str], ...] | None = None

        self._disposed = False
        self._subagent_roster_generation = 0
        self._subagent_roster_written_generation = 0
        self._subagent_roster_writer: asyncio.Task[None] | None = None
        # The CONTENT of the roster payload last written to the sidecar, so a
        # redundant write can be skipped. Live-only progress/output no longer
        # reach this hook, but defensive duplicate lifecycle signals and usage
        # updates can still leave the persisted projection ({version, jobs,
        # records}, minus the ever-incrementing ``generation`` counter)
        # byte-identical. A real
        # fan-out fired ~3 sidecar writes per settle, ~2 of them identical, and
        # each is a full mkstemp + fsync + os.replace + directory-fsync cycle;
        # that fsync volume is what tripped the macOS "disk writes exceeding
        # limit" throttle on a long delegating session. Mirrors the todo guard
        # (:meth:`_maybe_persist_todos`): fingerprint the durable content and
        # write only when it actually moved. ``None`` means nothing persisted
        # yet, so the first event always writes. The fingerprint itself is
        # computed ON THE WORKER THREAD by
        # :func:`_write_roster_sidecar_if_changed` — it is an O(roster)
        # ``json.dumps``, which must not run on the loop.
        self._persisted_roster_fingerprint: str | None = None
        # Session-scoped task group (HC-11): wake deliveries and aside
        # persistence are routed through it so dispose() cancels them
        # deterministically and a delivery after dispose never raises into an
        # unobserved task. A TaskGroup must be ENTERED inside a running loop,
        # so construction only allocates the slot — :meth:`async_init` opens
        # it (code that skips async_init degrades to ensure_future).
        self._tg_stack: contextlib.AsyncExitStack | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._signal: AbortSignal | None = None
        self._is_streaming = False
        #: Bang-mode receipts that completed while a turn or manual compaction
        #: owned the message list. They cannot be spliced into a live provider
        #: batch, but dropping them would make a visible command disappear on
        #: resume. The owner flushes this FIFO before releasing `_turn_lock`.
        self._pending_shell_records: list[tuple[str, ToolResult]] = []
        self._turn_lock = asyncio.Lock()  # serializes prompt() and wake deliveries
        # True only while an ON-DEMAND compaction holds ``_turn_lock``. The
        # automatic pass runs inside a turn and is covered by ``_is_streaming``;
        # this one runs between turns, so a caller that finds the lock held
        # needs a way to tell WHICH holder it is looking at (see `prompt`).
        self._compacting = False
        # Error text from the run just ended, journalled as a session incident
        # once persistence finishes (see _run_turn / journal_incident).
        self._pending_incident: str | None = None
        #: Journal notices (model switch / session incident) whose LIVE append
        #: arrived while a turn owned the message list. They are already on
        #: disk; only the live splice is parked, because appending one between
        #: an assistant's ``tool_calls`` and their ``tool_result``s builds a
        #: list no provider accepts and NOTHING repairs it afterwards, so the
        #: session re-sends the illegal prefix on every later turn and is
        #: bricked for good. Drained at the turn-safe boundaries (steering
        #: drain, pipeline exit, prompt entry, dispose) in the manner of
        #: ``_pending_shell_records``.
        self._pending_context_journal: list[CustomMessage] = []
        # (new_label, transient) of the last model switch made model-visible, so
        # the two edges that can both fire for one change (``set_model`` and a
        # route-settled event) do not double-announce. See journal_model_switch.
        self._last_model_switch_announced: tuple[str, bool] | None = None
        self._turn_task: asyncio.Task[None] | None = None  # in-flight turn (wake deliveries)

        # on_job_complete: settled model-owned jobs auto-deliver back into the
        # conversation when the session is idle (see _on_job_completed) — the
        # model stops having to poll 'jobs' for work it already started.
        #
        # max_running is operator-configurable (``values.subagents.max_running``)
        # because the right ceiling is a property of the machine and the models
        # in use, not of this code: a local model on a laptop wants far fewer
        # concurrent children than a hosted one. An unset or unusable config
        # contributes NO kwarg, so the manager's own default stands and the
        # behaviour is exactly what it was before this was configurable.
        def on_job_change() -> None:
            store = getattr(self, "_frontend_state_store", None)
            # No terminal or attach subscriber can observe these snapshots yet.
            # Avoid serializing large child trajectories solely for an unused
            # in-process state object; the first subscription snapshots live data.
            if store is None or not store.has_subscribers:
                return
            if getattr(self, "_frontend_jobs_refresh_scheduled", False):
                return
            self._frontend_jobs_refresh_scheduled = True
            try:
                # Child trajectory events arrive in bursts. A 50 ms coalesce
                # keeps progress perceptibly live without serializing six full
                # rosters on every shared-loop event.
                asyncio.get_running_loop().call_later(0.05, self._flush_frontend_jobs)
            except RuntimeError:
                self._frontend_jobs_refresh_scheduled = False
                store.refresh_jobs(self)

        self.jobs = AsyncJobManager(
            on_job_complete=self._on_job_completed,
            # One mutation seam persists resumability and publishes the live
            # canonical roster, including progress/cost/status changes.
            on_roster_change=self._schedule_subagent_persist,
            on_job_change=on_job_change,
            **_configured_max_running(),
        )
        self._wake = WakeScheduler(
            now=lambda: int(time.time() * 1000),
            # Deliveries route through the indirection hook: at resume the
            # session swaps in a catch-up shim that folds N overdue per-schedule
            # fires into one aggregated prompt (see _handle_missed_wakes).
            deliver=lambda due: self._wake_deliver_via_hook(due),
            persist=self._persist_wake_schedules,
            on_change=lambda: (
                self.refresh_frontend_state() if hasattr(self, "_frontend_state_store") else None
            ),
        )
        self._wake_deliver_hook: Callable[[DueWake], Awaitable[None]] = self._deliver_wake
        # Catch-up state for the wake deliveries the process was DOWN for:
        # (occurrences skipped while down, grace re-arm from the just-run
        # load(), text of the aggregated catch-up prompt, and whether that
        # prompt already went out). All four move together through
        # _handle_missed_wakes, which owns the lifecycle.
        self._missed_wake_occurrences: dict[str, int] = {}
        self._resume_grace_ends_ms: float = float("inf")
        self._resume_catchup_text: str | None = None
        self._resume_catchup_sent = False
        #: Ids of the overdue schedules the catch-up text folds. The shim
        #: swallows only these (see _deliver_wake_catchup); empty when there
        #: is no catch-up pending, so the shim is then a pure passthrough.
        self._resume_catchup_ids: set[str] = set()
        self._load_wake_schedules()
        # Rebuild this session's wake-index entry from the transcript on EVERY
        # open. The index (``local_operator.wakes.store``) is a derived file
        # that a supervisor and the picker read without opening the session;
        # rewriting it here is what makes it self-healing — a deleted, stale
        # or corrupt entry is repaired the next time the session is built,
        # and an empty schedule list removes it. Runs before the catch-up
        # snapshot only because ``load()`` has already re-armed overdue rows,
        # so what lands on disk is the post-load truth.
        self._rebuild_wake_index_entry()
        self._prepare_missed_wake_catchup()
        # The conversation's title is restored from the SAME transcript the
        # history came from, and for the same reason: a resumed session is the
        # same conversation, so the band and the terminal tab must name it the
        # way it was named before. Loaded here rather than by the host because
        # the transcript is the session's to read — every front end resuming a
        # session would otherwise need its own copy of this, and the one that
        # forgot would boot nameless.
        self._load_conversation_name()
        # The model the user explicitly SELECTED mid-session, restored before
        # the fallback pin below — deliberately in that order, because the pin's
        # own guard compares its persisted primary against the CURRENT
        # selection: a fallback that rescued the switched-to model must find
        # that model already restored, or the pin reads as stranded and is
        # dropped.
        self._restore_selected_model()
        # Subagents and todos: both live only in memory during a session, so a
        # resume rebuilds them from the transcript here — the job rows and comms
        # records that make the subagent panel and ``hub op='resume'`` work, and
        # the todo list the panel and the continuation guardrail read. Ordered
        # after the history replay (the transcript is already loaded) and before
        # the first turn (nothing has written to either store yet).
        self._load_subagent_roster()
        self._load_todo_snapshot()
        # Which model is ACTUALLY serving requests, when that is not the
        # selected one: the spec of the pinned fallback route, or None while
        # the primary serves. Owned by the session (not read off the stream
        # fn) because every reader — the TUI's model display, cost attribution,
        # persistence — needs it between events, and the stream fn is an
        # optional capability some hosts construct sessions without.
        self._active_fallback: ModelSpec | None = None
        # The pin itself — (selector, the chain entry's own effort or None) —
        # kept beside the derived spec above because the spec is a SNAPSHOT:
        # an `/effort` change while a fallback serves has to re-derive the
        # display spec from the pin, and the derivation's input is the target,
        # not the previous derivation.
        self._active_route: tuple[str, str | None] | None = None
        # AFTER the wake/name restores, same transcript, same reason: a
        # resumed session must come back on the model that was really
        # answering when it closed, not silently re-route the first prompt to
        # the provider that was failing.
        self._restore_active_route()
        # Same contract as the route restore, for the other half of what a
        # session was when it closed: the ``/team`` roster, ``/agent`` profile
        # and ``/goal`` ride the prompt's volatile tail, which is rebuilt empty
        # on every construction, so without this a resume dropped the persona
        # from the model's instructions entirely. Here rather than in
        # ``async_init`` because the tail must be populated before ANY turn can
        # be built, and quiet (no event) for the reason the route restore is:
        # nothing has subscribed yet, so hosts read the restored state when
        # they build their chrome.
        self._restore_attachment()
        # Owned here, not by the browser tool, for the same reason the wake
        # scheduler is: _build_tool_context runs at the start of EVERY turn, so
        # a handle the tool stored on the ToolContext lived exactly one turn.
        # The visible symptoms were that multi-turn browsing could not work at
        # all ("open X", then next message "click Y" → "no browser surface
        # open") and that every turn which opened a browser stranded a cmux tab
        # the agent could never close. dispose() closes whatever is still open.
        self._browser = BrowserSurface()

        # ``SESSION_CAPABILITY_TOOLS`` are createIf-gated on the ToolContext
        # fields only a SESSION can provide (subagent_launcher, jobs,
        # wake_scheduler, the ask hook). The factory that built this session
        # constructs its inventory from a context WITHOUT those fields, so those
        # tools silently never advertise — the model cannot delegate even
        # though the engine fully supports it (reproduced live: 144 requests,
        # 4M input tokens, zero task calls). Merge them in now that the
        # session's own context exists. This pass cannot cover a capability that
        # arrives LATER, which is why ``set_ask_handler`` re-runs it: the ask
        # hook is installed by the front end long after this returns.
        self._merge_capability_tools()
        # The session, not an OperatorApp, owns state that must survive a second
        # frontend joining. Constructed after every durable source above has
        # restored so the first snapshot is already authoritative.
        from local_operator.session.frontend_state import FrontendStateStore

        # Headless hosts restore the durable checkpoint WITHOUT the live source
        # scan (jobs/todos/MCP walk plus the TUI registry import) so schedulers
        # stay cheap — but never start bare: their turn-end checkpoint would
        # otherwise overwrite a TUI's persisted spend/duration/title.
        self._frontend_state_store = (
            FrontendStateStore.from_session(self)
            if self._has_ui
            else FrontendStateStore.from_checkpoint(self)
        )

    def _render_history(
        self, messages: list[AgentMessage], *, keep_images: bool = False
    ) -> list[Message]:
        """The configured transcript→LLM conversion, minus anything the active
        model cannot be sent.

        Every path that builds wire history goes through here rather than
        calling ``_convert_to_llm`` directly, because BOTH degrades have to
        hold for ALL of them. Compaction is the one that matters most: it has
        to send the history to summarise it, so a poisoned block makes even
        the escape hatch fail (anthropics/claude-code#50708) — and the same
        goes for a text-only model, whose refusal would otherwise brick the
        session exactly the way a provider refusal did.

        Two independent reasons strip images, checked in order of stickiness:

        - ``_images_rejected``: a provider REFUSED an image block, so the
          session never sends one again, whatever model it is on now.
        - the active spec's ``supports_images``: the model is KNOWN not to
          accept images (registry/discovery metadata), so the blocks are
          omitted instead of spending a request on a guaranteed refusal — the
          failure behind a session switched onto a text-only model while the
          history still carries screenshots. This one is NOT sticky: it reads
          the CURRENT spec, so switching back to a vision model restores the
          images on the very next render. ``keep_images=True`` suspends ONLY
          this strip (compaction's kept-window rebuild, which must not bake
          the omission into the live context); the sticky provider strip and
          the announcement still apply.

        Expired todo reminders are dropped here for the same reason: every path
        that reaches a provider has to be free of them.
        """
        rendered = self._convert_to_llm(self._live_todo_reminders(messages))
        # Immediately after the conversion and before any other pass, so EVERY
        # caller of this method is covered by one application: the turn path,
        # compaction, `_wire_legal_snapshot` and the token counter. Repairing at
        # the render rather than in a client fixes both wires at once — the
        # OpenAI Responses path emits a bare `role:user` item between
        # `function_call` and `function_call_output` and is rejected for the
        # same reason Anthropic rejects the coalesced form.
        rendered = _pair_spliced_tool_results(rendered)
        if self._images_rejected:
            # Nothing to rebound once images are being dropped outright, and
            # dropping first saves decoding a block that is about to become a
            # one-line notice.
            return _without_images(rendered)
        if not self._model.supports_images and not keep_images:
            # ``rendered`` is the PRE-strip view here: exactly what the
            # omission is about to take away, and what the announcement
            # reports on.
            self._announce_text_only_omission_once(rendered)
            return _without_images(rendered, model_incapable=True)
        return _rebound_history_images(rendered)

    def _announce_text_only_omission_once(self, rendered: list[Message] | None = None) -> None:
        """Say, once per session, that the active model is not seeing the
        conversation's images.

        Fires from the render rather than ONLY from ``set_model`` so EVERY
        path that strips images explains itself: a mid-session switch, a
        resume booted on a text-only default (the boot render runs in the
        TUI's preloaded-context measurement, before any turn), a new image
        attached after the switch. A switch-only notice left those paths
        silent — a user who opened yesterday's screenshot-heavy session under
        a new text-only default got answers that ignored the screenshots and
        no line saying why (design review round 1, D1). ``set_model`` ALSO
        calls this on a vision→text-only flip so a switch explains itself
        immediately, without waiting for the next render.

        The latch makes it once-per-session, not once-per-render: the render
        runs on every provider call, and a repeating warning is noise. The
        omission itself stays non-sticky — it reads the current spec on every
        render — so switching to a vision model restores the images without
        touching this latch.

        ``_spawn_background`` because ``set_model`` is sync and the
        announcement is advisory: the render already applies the omission, so
        a notice that never runs costs nothing but the log line.
        """
        if self._text_only_omission_announced:
            return
        if rendered is None:
            # Presence check only: the plain convert output already carries
            # every image block, and the rebound pass would spend a decode
            # (and a Pillow re-encode for any oversized block) on a question
            # the header sniff cannot answer. ``set_model`` calls this on the
            # TUI event loop; a paste-heavy history must not stall a keypress.
            rendered = self._convert_to_llm(self._live_todo_reminders(list(self._context.messages)))
        if not any(
            isinstance(block, ImageContent) for message in rendered for block in message.content
        ):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop to carry the notice (a render from a sync caller, e.g.
            # construction-time configuration). The omission still applies;
            # only the announcement is skipped — and the latch stays unset so
            # a later render WITH a loop can still announce.
            return
        self._text_only_omission_announced = True
        # ``model_label`` — the provider/model_id vocabulary the status band
        # and the switch receipt already use — rather than the display name:
        # two names for the same object two lines apart read as two models
        # (design review round 1, D3).
        label = self.model_label
        logger.warning(
            "model %s does not accept images; omitting them from this session's "
            "context while it is active (%s)",
            label,
            self._image_drop_diagnostic(),
        )
        self._spawn_background(
            self._emit(
                NoticeEvent(
                    text=(
                        f"{label} does not accept images — the images in this "
                        "conversation are omitted from its context. Switch to a "
                        "model that accepts images to see them again."
                    ),
                    kind="warning",
                )
            )
        )

    def _live_todo_reminders(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        """``messages`` without todo reminders the list has since outrun.

        A reminder is a POINT-IN-TIME assertion — "these items are still open" —
        so it expires the instant any item changes status. It lives on in
        ``_context.messages`` (follow-ups are never removed from the live list),
        and once the model has done the work, replaying it keeps insisting that
        finished or blocked work is still open. That is a lie the model then has
        to spend a turn arguing with, and it is the same staleness the renderer's
        newest-only rule guards against for TWO reminders; this covers the one
        reminder that simply went out of date.

        It lives on the session rather than in :func:`_default_convert_to_llm`
        because staleness needs the session id, and that function must stay a
        pure function of its message list (``session_factory`` aliases it as the
        default renderer for hosts that pass none).

        Costs one scan and nothing else in the overwhelmingly common case: no
        reminder in the list means the original list is handed straight back,
        without even reading the store.
        """
        if not any(_is_todo_reminder(message) for message in messages):
            return messages
        current = todo_fingerprint(self._session_id)

        def expired(message: AgentMessage) -> bool:
            return (
                _is_todo_reminder(message) and _stamped_todo_fingerprint(message.details) != current
            )

        return [message for message in messages if not expired(message)]

    def _render_for_compaction(self, *, keep_images: bool = False) -> list[Message]:
        """The rendered history a compaction pass plans and commits against.

        :meth:`_render_history` minus the todo reminders, because a reminder is
        the ONE injection nothing persists (``_todo_continuation`` hands it to
        the loop as a follow-up, which emits no event and reaches no
        transcript), and compaction is built on the rendered history being
        persisted history. Rendered into it, one reminder broke the pass at both
        ends:

        - ``_plan_compaction``'s replayability guard matches ``kept[0].id``
          against the transcript's entry ids, and a reminder's id is in no
          transcript. A reminder AT the cut therefore refused the entire pass as
          ``cut_not_replayable`` — measured at 30/30 refusals on a session with
          one open todo against 25/30 committed with none — and it recurred
          rather than self-healing, because the next turn puts a new reminder at
          the same structural offset. The automatic gate has nobody to tell, so
          the symptom was a session that silently stopped compacting.
        - ``_run_compaction`` rebuilds the context from THIS history, so a
          reminder inside the kept window came back as a plain
          ``Message(role="user")``. Both guards that expire a nudge
          (:meth:`_live_todo_reminders` and the newest-only rule in
          :func:`_default_convert_to_llm`) match ``CustomMessage`` only, so the
          baked copy was invisible to both and went on asserting "these todo
          items are still open" for the rest of the session — after the items
          were done, and beside a newer live reminder.

        Excluded rather than rescued (advancing the cut past it, or re-attaching
        it after the commit) because the reminder is EPHEMERAL by design, and
        both rescues ask a transient message to be something it never was — a
        replayable anchor, or a message that survives a rebuild it is not
        persisted for. Nothing is lost by dropping it: no compaction pass can
        run before the model has already read the nudge (``on_turn_end`` fires
        only when the loop will CONTINUE, i.e. after the model answered it with
        tool calls; the post-turn gate and ``/compact`` run later still), a
        resume drops it anyway, and the guardrail re-arms on the next list
        movement or user turn.

        A ``session_credential`` record is filtered for the SAME reason (review
        round 1, R2): it is not persisted (see
        :data:`_PERSISTABLE_CUSTOM_TYPES`), but unlike a reminder it sits at
        the RECENT end of the context — exactly where the kept window is — so
        the default render would bake it into ``kept`` as a plain
        ``Message(role="user")`` carrying the same id, and the turn-end
        persist pass (which persists every plain Message) would then write it
        to the transcript after all, resurrecting the stale-on-resume replay
        the allow-list removal closed. Excluding it here keeps the live
        announcement in the REQUEST render (``_render_history``, untouched)
        while the rebuild the compaction commit owns stays persisted-only.
        """

        def _is_ephemeral_for_compaction(message: AgentMessage) -> bool:
            """Whether ``message`` must stay out of the compaction render.

            Named rather than inlined because BOTH filters are
            live-context-only injections whose rendered (id-carrying) copy
            would otherwise be baked into the kept window and persisted by the
            turn-end pass despite never being transcript material.
            """
            return _is_todo_reminder(message) or (
                isinstance(message, CustomMessage)
                and message.custom_type == SESSION_CREDENTIAL_MESSAGE_TYPE
            )

        return self._render_history(
            [
                message
                for message in self._context.messages
                if not _is_ephemeral_for_compaction(message)
            ],
            keep_images=keep_images,
        )

    async def _degrade_if_image_rejected(self, error: BaseException | str) -> None:
        """Stop sending images if that is what the provider just refused.

        Idempotent and one-way. The block is in the HISTORY, so without this
        every later request re-sends it and fails identically — reload replays
        it, and ``/compact`` cannot run either because summarising means
        sending the history first. That is the reported symptom: a session that
        answers every prompt, forever, with the same 400.

        Not a preventable condition on our side. Anthropic accept the same
        bytes for hours and then start refusing them
        (anthropics/claude-code#50708), so validating on the way in cannot
        close it; the only defence is to notice and stop.

        The turn that discovered this still fails — the request is already
        spent, and retrying it here would mean re-entering the loop from inside
        its own end event, past the boundary bookkeeping this branch has
        already had to repair twice. The next turn, and ``/reload``, both
        succeed, which is the difference between a session that recovers and
        one that is finished.
        """
        # Imported HERE, not at module scope: this module is the CLI's
        # composition root and docs/REWRITE.md forbids module-level provider
        # imports, because they are what makes importing a session expensive
        # (and, done at the top of this file, cyclic).
        from local_operator.providers.failover import is_image_rejection

        if self._images_rejected or not is_image_rejection(error):
            return
        self._images_rejected = True
        logger.warning(
            "provider rejected an image; dropping images from this session's context (%s)",
            self._image_drop_diagnostic(),
        )
        await self._emit(
            NoticeEvent(
                text=(
                    "The provider rejected an image in this conversation's history. "
                    "Images have been dropped from the context so the session keeps "
                    "working — send your message again."
                ),
                kind="warning",
            )
        )

    def _image_drop_diagnostic(self) -> str:
        """Structure of the images this degrade is about to drop, for the log.

        The backstop above is deliberately blind to WHY the provider refused,
        which once turned a serialization defect of ours into a silent,
        permanent loss of a whole compacted history — with a receipt still
        claiming "82% smaller", because ``compaction/tokens.py`` charges a flat
        estimate per image block without looking at ``data``. The first frame's
        decoded magic separates the two causes on sight: ``89504e47`` is a real
        PNG the provider refused, ``6956424f`` (ASCII ``iVBO``) is base64 we
        encoded twice. Structure only — never message text, never the payload.

        ``_convert_to_llm`` directly, not ``_render_history``: the flag is
        already set by the time this runs, so the rendered history no longer
        contains the blocks being reported on.
        """
        try:
            images = [
                block
                for message in self._convert_to_llm(list(self._context.messages))
                for block in message.content
                if isinstance(block, ImageContent)
            ]
            if not images:
                return "no image blocks in rendered history"
            first = images[0]
            magic = base64.b64decode(first.data, validate=True)[:4].hex()
            return (
                f"{len(images)} image block(s); first: mime={first.mime_type} "
                f"b64_len={len(first.data)} magic={magic}"
            )
        except Exception as exc:  # noqa: BLE001 — a diagnostic must never break the degrade
            return f"diagnostic unavailable: {exc!r}"

    def _merge_capability_tools(self, names: Sequence[str] = SESSION_CAPABILITY_TOOLS) -> None:
        """Add the tools gated on this session's own capabilities.

        ``create_tools`` is createIf-driven: a builder returns ``None`` unless
        the ToolContext carries what the tool needs. The factory context has
        no ``subagent_launcher``/``jobs``/``wake_scheduler`` (they are
        session-owned), so ``task``/``wait``/``jobs`` (and any future
        session-gated tool) never reached the inventory. Re-running the same
        builders against ``self._build_tool_context()`` yields exactly those
        tools; merge them (replacing same-named placeholders) so the model's
        tool surface reflects what this session can actually do.

        ``names`` exists because this merge is ALSO the rescue for a capability
        that arrives after construction, and those must not rescue each other:
        :meth:`set_ask_handler` merges ``("ask",)`` alone, so installing a
        question surface cannot resurrect a ``wake`` or ``task`` tool that
        ``_build_child_session`` deliberately pruned from a subagent's
        inventory.
        """
        try:
            from local_operator.tools.registry import create_tools

            capability = create_tools(self._build_tool_context(), enabled=names)
        except Exception:  # tooling must never break session construction
            return
        if not capability:
            return
        by_name = {tool.name: tool for tool in capability}
        merged = list(self._tools)
        seen = {tool.name for tool in merged}
        for name, tool in by_name.items():
            if name in seen:
                merged = [tool if t.name == name else t for t in merged]
            else:
                merged.append(tool)
        self._tools = merged
        self._context.tools = merged

    async def async_init(self) -> None:
        """Async second half of construction.

        Opens the session-scoped task group and re-arms the wake scheduler:
        a scheduler armed during sync ``__init__`` (no running loop yet)
        could not create its timer — one ``pump()`` here fires overdue wakes
        and arms properly (see ``WakeScheduler.needs_rearm``). Overdue wakes
        adopted from a previous session are first folded into a single
        catch-up prompt (``_handle_missed_wakes``) so a resume costs one
        wake turn, not one per overdue schedule. Safe to call more than
        once; sessions that skip it degrade to ``ensure_future``
        for background work.
        """
        if self._tg_stack is None and not self._disposed:
            stack = contextlib.AsyncExitStack()
            await stack.__aenter__()
            try:
                self._task_group = await stack.enter_async_context(asyncio.TaskGroup())
            except BaseException:
                await stack.aclose()
                raise
            self._tg_stack = stack
        self._handle_missed_wakes()
        await self._wake.pump()

    # -- identity / state (SessionProtocol) ----------------------------------
    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def subagent_comms(self) -> Any:
        """This session's channel to the subagents it launches.

        Minted on first use rather than in ``__init__`` so a child — handed
        its PARENT's instance at construction — never creates a second one
        that nobody is listening to, and so a session that never delegates
        pays nothing for the capability.
        """
        if self._subagent_comms is None:
            self._subagent_comms = SubagentComms(self)
        return self._subagent_comms

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def model_label(self) -> str:
        return f"{self._model.provider}/{self._model.model_id}"

    def _system_blocks(self, model: ModelSpec | None = None) -> list[str] | Awaitable[list[str]]:
        """Invoke the block provider with one internally consistent model label.

        The provider gained an optional ``model_label`` argument so the env
        block can name the running model (see ``build_system_blocks``). A
        provider that predates the argument — a bare zero-arg callable a host or
        a test supplies — is still valid: its arity is inspected once (not
        probed by catching ``TypeError``, which would mask a genuine TypeError
        raised INSIDE a one-arg provider and re-invoke it) and it is called with
        no argument. The label is therefore strictly additive: no caller is
        forced to grow a parameter it does not use. ``model`` is supplied by the
        loop before an async block build so that build and the request use the
        same snapshot; other callers omit it and read the session's live model.
        """
        model_label = (
            f"{model.provider}/{model.model_id}" if model is not None else self.model_label
        )
        if not self._blocks_provider_takes_label:
            return self._system_blocks_provider()
        if self._blocks_provider_arity_certain:
            # Signature was readable and takes the label: any TypeError from here
            # is a genuine bug inside the provider and must surface, not be
            # swallowed by a zero-arg retry.
            return self._system_blocks_provider(model_label)
        try:
            return self._system_blocks_provider(model_label)
        except TypeError:
            # Unreadable signature guessed one-arg and guessed wrong: this
            # provider is genuinely zero-arg. Remember it and call the
            # compatible way from now on.
            self._blocks_provider_takes_label = False
            return self._system_blocks_provider()

    @property
    def model(self) -> ModelSpec:
        """The spec every provider call is built from."""
        return self._model

    @property
    def active_fallback(self) -> ModelSpec | None:
        """The pinned fallback's spec while one is serving requests, else None."""
        return self._active_fallback

    @property
    def routing_settings(self) -> Any:
        """The settings this session's stream will ROUTE on, or ``None``.

        The mapping the stream holds — captured at build and rebound by the
        config watcher through :meth:`_apply_config_change` on every change —
        so it IS what the session will do next. Kept as a pass-through to the
        stream (rather than a second copy stored here) so the two can never
        disagree; ``None`` when the stream predates the accessor, which callers
        must treat as "cannot say", never as "empty".
        """
        return getattr(self._stream_fn, "routing_settings", None)

    @property
    def effective_model(self) -> ModelSpec:
        """The spec ACTUALLY serving requests.

        ``model`` is what the user selected; while a provider fallback is
        pinned the requests go elsewhere, and a display reading ``model``
        asserts a model that is not answering. Front ends paint from THIS.
        """
        return self._active_fallback if self._active_fallback is not None else self._model

    @property
    def effective_model_label(self) -> str:
        """``provider/model`` of the spec actually serving requests."""
        spec = self.effective_model
        return f"{spec.provider}/{spec.model_id}"

    def has_admitted_command(self, command_id: str) -> bool:
        """Return whether a producer command is durably in this transcript."""
        return self._transcript.has_admitted_command(command_id)

    def subscribe_admitted_commands(self, handler: Callable[[str], None]) -> Callable[[], None]:
        """Observe command IDs once their transcript append has succeeded."""
        return self._transcript.subscribe_admitted_commands(handler)

    def subscribe_rejected_steering(
        self, handler: Callable[[str, str], None]
    ) -> Callable[[], None]:
        """Observe producer steers that failed before durable admission."""
        self._steering_rejection_handlers.append(handler)

        def unsubscribe() -> None:
            try:
                self._steering_rejection_handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def history(self) -> list[AgentMessage]:
        """The conversation as replayed into LLM context, in order.

        Read-only for rendering: a front end that boots against a resumed
        session wants the prior conversation back on screen, and the source of
        truth for what the model sees is ``self._context.messages`` (populated
        at construction from ``transcript.build_llm_history()``). Returning the
        live list is cheap and always in sync with what the loop will act on.
        Callers must treat it as immutable — mutating a message here would
        corrupt the compaction token cache for the same object (see the
        ``Message`` docstring).
        """
        return list(self._context.messages)

    def context_breakdown(self) -> dict[str, int]:
        """On-demand token breakdown for the context the next request sends.

        This is the user-visible counterpart to the 30k start-context budget:
        it tokenizes the actual system blocks + wire tool schemas + rendered
        messages, so a 126-tool MCP server's cost is a fact the user can SEE
        instead of an invisible latency/cost regression. The four fixed blocks
        retain their cache-layout names; schemas are counted separately
        because providers serialize them beside the prompt, not in block 1.
        """
        import json

        from local_operator.compaction.tokens import (
            approx_text_tokens,
            estimate_messages_tokens,
        )

        blocks = list(self._context.system_blocks)
        while len(blocks) < 4:
            blocks.append("")
        result = {
            "instructions": approx_text_tokens(blocks[0]),
            "tool_inventory": approx_text_tokens(blocks[1]),
            "environment": approx_text_tokens(blocks[2]),
            "knowledge_mcp_goal": approx_text_tokens(blocks[3]),
            "tool_schemas": sum(
                approx_text_tokens(
                    tool.name
                    + "\n"
                    + (tool.description or "")
                    + "\n"
                    + json.dumps(tool.parameters, sort_keys=True, separators=(",", ":"))
                )
                for tool in self._tools
            ),
            "messages": estimate_messages_tokens(
                self._render_history(list(self._context.messages))
            ),
            # The EFFECTIVE model's window: the breakdown predicts whether the
            # NEXT request fits, and while a fallback serves that request goes
            # to the fallback — measuring it against the selected model's
            # window misstates the one number the panel exists to report.
            "context_window": int(self.effective_model.context_window),
            "cache_read": int(
                self._last_usage.cache_read_tokens if self._last_usage is not None else 0
            ),
        }
        result["total"] = sum(
            result[key]
            for key in (
                "instructions",
                "tool_inventory",
                "environment",
                "knowledge_mcp_goal",
                "tool_schemas",
                "messages",
            )
        )
        return result

    def set_model(self, model: ModelSpec, *, explicit: bool = False) -> None:
        """Swap the model spec, in force from the very next provider call.

        Not "from the next turn": the running turn picks it up too. A turn is a
        chain of provider calls with tool batches between them, and the loop
        re-reads this spec at each of them (``LoopConfig.get_model``), so a
        switch made while the agent is working lands at the next call boundary.
        Anything already in flight finishes on the spec it was issued with — the
        switch never splits one response across two models.

        That boundary is the whole point of the feature. A user reaches for
        ``/model`` mid-turn BECAUSE the running model is doing badly, and the
        old "next turn" behaviour meant the switch was invisible for the rest of
        a long tool-using turn: the band said the new model, every remaining
        call went to the old one.

        Also used for per-request overrides that are not part of the agent
        record — the FastAPI server applies ``ChatRequest.options``
        (temperature / top_p) this way, since sampling rides on the spec (see
        ``model/configure.build_model_spec``).

        Tells the stream fn the model changed, so state frozen per user message
        against the OLD model — the auto-effort level, the quota preflight's
        route — is re-fitted to the new one rather than carried across a switch
        it was never computed for. That hook must be SYNCHRONOUS: this method is
        sync and would discard a returned coroutine, which is the right contract
        for what is only a cache re-fit.

        ``explicit`` marks a deliberate model CHOICE — the ``/model`` command and
        the phone's model switch — as opposed to an internal knob adjustment
        (``/effort``, per-request sampling overrides) that rewrites the spec
        without choosing a model. The distinction only matters while a provider
        fallback is pinned: an explicit choice withdraws the pin even when it
        re-selects the very model the fallback displaced, because that is the
        user reclaiming it. A knob change never withdraws — the user is adjusting
        the model that is serving them, fallback or not.
        """
        previous = self._model
        self._model = model
        if previous.supports_images and not model.supports_images:
            # A genuine move onto a text-only model (same-model knob changes
            # copy the spec and cannot flip the capability): explain the
            # omission now, through the same once-per-session latch the
            # render uses, instead of waiting for the next render to say it.
            self._announce_text_only_omission_once()
        if (previous.provider, previous.model_id) == (model.provider, model.model_id):
            if explicit and self._active_fallback is not None:
                # An explicit re-selection of the model a fallback displaced: the
                # user is reclaiming it, so the pin's premise is withdrawn even
                # though the selector did not move. The stream fn's route clear is
                # selector-driven (preflight resets on a NEW selector), and a
                # same-model re-selection never changes the selector — so that
                # lazy clear would never fire and the session would stay glued to
                # the fallback until the user switched away and back. Tell the
                # stream fn directly.
                self._drop_fallback_pin(model, "model reselected")
                withdraw = getattr(self._stream_fn, "withdraw_fallback", None)
                if callable(withdraw):
                    withdraw()
                self._journal_effort_if_selection_in_force(previous, model)
                self.refresh_frontend_state()
                return
            # Same model, different knobs (effort, sampling): nothing routing
            # or quota related has moved, so leave the frozen per-message state
            # alone. `/effort` and the server's option overrides take this path
            # on every call and must not each cost a re-fit.
            #
            # `base_url` is deliberately NOT part of the key. It is derived from
            # the provider definition rather than chosen per call, so it cannot
            # vary independently of the pair above; and the state this guards is
            # itself selector-keyed (`SessionStreamFn._primary_selector`), so a
            # third component here would invalidate more often than the thing
            # being invalidated can actually change.
            #
            # One exception rides this path: while a fallback serves, the
            # DERIVED display spec carried the previous effort onto the target
            # (see `spec_for_target` — a chain entry naming no effort inherits
            # the chosen level), so an `/effort` change has to re-derive it or
            # the band keeps naming the level the user just moved away from.
            # Quietly — no event, no persistence: the route did not move, and
            # the `/effort` receipt plus the host's own repaint already cover
            # the level change.
            if self._active_route is not None:
                refreshed = self._spec_for_route(*self._active_route)
                if refreshed is not None:
                    self._active_fallback = refreshed
            self._journal_effort_if_selection_in_force(previous, model)
            self.refresh_frontend_state()
            return
        # A genuine model change is a SELECTION, and selections outlive the
        # process: journal it so `--resume` comes back on this model instead
        # of the boot default (see SELECTED_MODEL_CUSTOM_TYPE). Every knob
        # change (`/effort`, the server's sampling overrides) took the
        # same-pair early return above, so only real switches reach this line
        # and the journal stays one row per switch, not one per keystroke.
        self._selected_model_dirty = True
        self._spawn_selected_model_write()
        # An explicit switch withdraws the fallback pin's premise: the pin
        # rescued the PREVIOUS selection, and the stream fn's preflight will
        # clear its own route state the moment it sees the new selector. The
        # display state has to move in the same step — a band still naming the
        # old fallback after `/model` is the same stale frame this state
        # exists to prevent. Persisted (with the new primary) so a resume does
        # not restore a pin the user already switched away from.
        if self._active_fallback is not None:
            self._drop_fallback_pin(model, "model switched")
        notify = getattr(self._stream_fn, "on_model_changed", None)
        if callable(notify):
            notify(model)
        # Make the deliberate switch visible to the MODEL, not just the host's
        # band. A same-pair knob change never reaches here (it took an early
        # return above), so this fires once per genuine switch. Background
        # because ``set_model`` is sync and runs on the UI loop; the env-block
        # ``Model:`` line keeps identity correct even if this notice slips a
        # turn (see journal_model_switch). No ``reason``: the head already reads
        # "now running as X (was Y)", so a "Reason: model switched" line would
        # only repeat it. ``reason`` is reserved for failover causes (R3).
        self.refresh_frontend_state()
        self._spawn_background(
            self.journal_model_switch(
                f"{model.provider}/{model.model_id}",
                f"{previous.provider}/{previous.model_id}",
                transient=False,
            )
        )

    def _journal_effort_if_selection_in_force(self, previous: ModelSpec, model: ModelSpec) -> None:
        """Re-journal the selection when a knob change moves the effort on a
        model the user SWITCHED to, so the restored effort tracks it.

        The ``selected_model`` row carries an effort field, and a switch
        followed by a SEPARATE ``/effort`` change would otherwise leave that
        field frozen at the switch's level: the ``/effort`` call takes
        ``set_model``'s same-pair early return and never reaches the journal
        write, so a resume would come back on the switched model at its
        DEFAULT effort — silently dropping the level the user chose and, on a
        reasoning model, quietly changing cost (review round 1, F1).

        Bounded twice, and each bound is load-bearing:

        - **Only an actual effort move.** A sampling-only override
          (temperature / top_p, the server's ``ChatRequest.options`` path)
          rides ``set_model`` on the same pair too, and the journal stores no
          sampling — so re-journalling on those would append identical rows for
          a change the row cannot even express.
        - **Only a non-boot selection.** The boot model deliberately persists
          no effort (the ``/effort`` non-goal): ``_restore_selected_model``
          skips a row whose selector equals the boot selector, so a boot-model
          effort change has nothing to restore onto and journalling one would
          both be inert and break the non-goal. A selection the user switched
          to already has a row whose effort must stay truthful.
        """
        if previous.reasoning_effort == model.reasoning_effort:
            return
        if f"{model.provider}/{model.model_id}" == self._boot_selector:
            return
        self._selected_model_dirty = True
        self._spawn_selected_model_write()

    def _drop_fallback_pin(self, primary: ModelSpec, reason: str) -> None:
        """Withdraw the pinned fallback: clear display state, persist, announce.

        Shared by the two withdrawal edges — a switch to a DIFFERENT model and an
        explicit re-selection of the SAME model — because both are "the user
        chose a model, so the rescue route stops speaking for them". Persisted
        with the new primary so a resume does not restore a pin the user already
        withdrew; the ``ModelChangeEvent`` repaints any host display still
        naming the fallback.
        """
        self._active_fallback = None
        self._active_route = None
        self._spawn_background(self._persist_active_route(primary))
        self._spawn_background(
            self._emit(
                ModelChangeEvent(
                    provider=primary.provider,
                    model_id=primary.model_id,
                    effort=primary.reasoning_effort,
                    reason=reason,
                    is_fallback=False,
                    context_window=int(primary.context_window),
                )
            )
        )

    @property
    def goal(self) -> str:
        """The session's standing objective ("" when unset)."""
        return self._goal_state.text

    @property
    def agent_brief(self) -> str:
        """The instructions ``/agent`` last stamped on the tail ("" when none).

        Exposed so the TUI can tell an attach that actually layered a persona
        apart from one that resolved a real NAME but carried no instructions
        (A2): the two cases deserve different notices, and the front end has no
        other window onto the volatile tail.
        """
        return self._goal_state.agent_brief

    @property
    def active_agent(self) -> str:
        """The DISPLAY NAME of the ``/agent`` profile in force ("" when none).

        Distinct from :attr:`agent_brief`, which is the opaque instruction blob:
        the band's active-profile segment (U2) needs to NAME the persona, and a
        role that resolved but stamped nothing (A2) has an empty brief yet still
        counts as attached, so the brief alone cannot answer "which profile".
        Mirrors how ``agent_brief`` is surfaced — a read-only view onto the
        volatile tail's holder — so the front end never reaches into
        ``_goal_state`` for it.
        """
        return self._goal_state.agent_name

    def set_goal(self, text: str) -> str:
        """Set (or clear, with an empty string) the standing objective.

        Returns what was actually stored (trimmed and length-capped). The goal
        rides the system prompt's volatile tail, which is re-read before every
        provider call: idle changes apply to the next turn, and mid-turn changes
        apply to the next model step. Only that tail changes — never the cached
        prefix or an in-flight request.
        """
        stored = self._goal_state.set(text)
        # Same tail, same fate on resume as the team/agent briefs, so the goal
        # is journalled by the same mechanism rather than a second one.
        self._persist_attachment()
        self.refresh_frontend_state()
        return stored

    @property
    def active_team_name(self) -> str:
        """The NAME of the team this session manages ("" when none).

        The band's active-team segment (U2) names the roster in force. Read off
        ``active_team.name`` rather than a separate field because the team object
        is already the source of truth and can carry no other name; guarded so a
        detach (``active_team is None``) or a nameless test double both resolve
        to "" and the segment simply disappears.
        """
        team = self.active_team
        if team is None:
            return ""
        return str(getattr(team, "name", "") or "")

    def attach_team(self, team: Any) -> None:
        """Bind this session as the manager of ``team``.

        The team's collaboration and project briefs ride the volatile tail
        (see ``build_system_blocks``) so they apply from the next turn
        without rebuilding the session or invalidating the cached persona
        prefix. Children spawned after this inherit the same team via
        :attr:`active_team`.
        """
        self.active_team = team
        # Either branch is the user acting on the team slot, so a carried
        # unresolved name stops being a recovery hint here (R1): a detach means
        # they no longer want it, and an attach replaces it outright.
        self._clear_unresolved("team")
        if team is None:
            self._goal_state.team_brief = ""
            self._persist_attachment()
            self.refresh_frontend_state()
            return
        preamble = getattr(team, "manager_preamble", lambda: "")()
        # The manager's own profile instructions (the reusable BASE) sit in
        # front of the team brief so a custom manager keeps its voice when
        # this session was not launched with --agent. A packaged starter
        # resolved by name is enough; a missing profile costs nothing.
        manager_name = getattr(team, "manager", "") or ""
        if manager_name and self.agent_registry is not None:
            # Same resolution ORDER as ``/agent`` attach, via the ONE shared
            # resolver: own role, then own specialist (BEFORE a same-named
            # packaged seed), then the seed. This closes the twin of A1 — a
            # team whose manager is the operator's own specialist named after a
            # seed word (reviewer/coder/…) previously layered the SEED here,
            # because ``resolve_profile`` was consulted before the specialist
            # path. Both call sites share ``_resolve_profile_or_specialist`` so
            # the order cannot drift between them again.
            kind, profile, specialist_prompt, _ = self._resolve_profile_or_specialist(manager_name)
            if kind in ("role", "seed") and profile is not None and profile.preamble:
                preamble = profile.preamble + (preamble or "")
            elif kind == "specialist" and specialist_prompt:
                # A specialist has no role preamble header; wrap it so the model
                # can tell the manager's own base voice apart from the team brief
                # layered after it.
                preamble = (
                    "<manager-profile>\n"
                    + specialist_prompt
                    + "\n</manager-profile>\n\n"
                    + (preamble or "")
                )
        self._goal_state.team_brief = preamble or ""
        # Journal the NAME so a resume can rebuild this tail. On change only:
        # the roster moves a handful of times per session, and the brief itself
        # is deliberately not stored (see ``SessionAttachment``).
        self._persist_attachment()
        self.refresh_frontend_state()

    def _persist_attachment(self) -> None:
        """Journal the attached team/agent/goal beside the transcript.

        Called from every mutation of the attached identity — ``attach_team``
        (from BOTH of its branches: the ``team is None`` detach returns early,
        so the tail write cannot cover it), ``_stamp_agent_brief``,
        ``clear_agent_profile`` and ``set_goal`` — because those are the only
        ways the volatile tail's persistent half moves. Writing from the
        mutators rather than from the TUI commands is what makes this hold for
        every front end: the mobile relay sets a goal through
        ``Session.set_goal`` too, and a front-end-side write would have left
        that path unpersisted.

        Synchronous and best-effort. The write is a sub-kilobyte atomic replace
        and these mutators are not on the hot turn path, so a thread hop would
        buy nothing and would open a window where a session disposed
        immediately after ``/team`` lost the attachment it just reported. The
        helper never raises by contract (see ``write_session_attachment``), and
        the broad guard here covers a reduced test double whose transcript has
        no directory rather than any failure of the write itself.

        A name the restore could not resolve is written back UNCHANGED rather
        than as the empty live value (R1). Without that fallback a transient
        miss survived only until the next mutation: ``active_team`` is ``None``
        for the half that failed, so a plain ``/goal`` in the resumed session
        journalled the emptiness over the stored name and the attachment was
        lost for good — one command after a failure the design calls
        recoverable. The carried name is dropped only when the user acts on
        that slot themselves (see :meth:`_clear_unresolved`), so "recoverable"
        holds for the whole life of the session rather than for the duration of
        the restore.
        """
        from local_operator.resume import write_session_attachment

        if self._restoring_attachment:
            return
        try:
            directory = self._transcript.directory
        except Exception:  # noqa: BLE001 — a reduced host must not lose its turn
            return
        write_session_attachment(
            directory,
            team=self.active_team_name or self._unresolved_team,
            agent=self._goal_state.agent_name or self._unresolved_agent,
            goal=self._goal_state.text,
        )

    def _clear_unresolved(self, slot: str) -> None:
        """Forget a carried unresolved name because the user acted on that slot.

        The counterpart to the fallback in :meth:`_persist_attachment`. A
        carried name is a RECOVERY hint for a team or profile that was
        momentarily unreachable, so it must survive incidental mutations — but
        it must NOT survive the user deliberately attaching something else or
        detaching, or a ``/team other`` would leave the old name to reappear at
        the next resume and re-attach a roster the user had moved off.

        Per slot, because the two are independent: re-attaching an agent says
        nothing about whether the stored team is still wanted.
        """
        if slot == "team":
            self._unresolved_team = ""
        else:
            self._unresolved_agent = ""

    def _restore_attachment(self) -> None:
        """Re-attach the team/agent/goal this session was carrying when it closed.

        THE resume fix. The team and agent briefs live only in ``GoalState``,
        which ``session_factory`` builds empty on every construction, so before
        this a ``--resume`` reopened the conversation with the persona GONE
        from the system prompt — not merely missing from the status band. The
        band was reporting the truth, which is why it must keep being driven
        from the session (see the TUI's ``_sync_team_band``) and never painted
        from the sidecar directly: a segment naming a team that failed to
        resolve would turn an honest blank into a lie.

        Re-resolves by NAME through the same entry points the live ``/team``
        and ``/agent`` commands use (``attach_team`` after a registry lookup,
        ``attach_agent_profile``, and through it the one shared
        ``_resolve_profile_or_specialist`` ordering). Deliberate, and the
        reason the sidecar stores names instead of briefs: the operator may
        have edited the team's briefs or the profile's instructions since the
        session last ran, and a stored brief would resume them onto a
        definition that no longer exists. Going through the live resolvers also
        means this path cannot drift from the attach path — a change to
        resolution order applies to a resumed session automatically.

        Runs during construction, BEFORE any turn, so the restored briefs are
        in the tail the first prompt is built from. It mutates the shared
        ``GoalState`` holder the system-blocks provider already closed over, so
        nothing is rebuilt and the cached persona PREFIX is untouched — only
        the volatile tail this state has always lived in.

        Failures degrade to unattached and are RECORDED for the host to report
        (:attr:`attachment_restore_notice`) rather than raised: a team the user
        renamed or deleted must not make a conversation unopenable.
        """
        from local_operator.resume import read_session_attachment

        try:
            stored = read_session_attachment(self._transcript.directory)
        except Exception:  # noqa: BLE001 — a resume must always open
            return
        if stored is None:
            return

        self._restoring_attachment = True
        try:
            self._apply_stored_attachment(stored)
        finally:
            self._restoring_attachment = False

    def _apply_stored_attachment(self, stored: "SessionAttachment") -> None:
        """Re-apply one parsed attachment; see :meth:`_restore_attachment`.

        Split out so the suppression flag around it is a plain ``try/finally``
        with no early ``return`` able to skip the reset.
        """
        if stored.goal:
            # Straight onto the holder: ``set_goal`` would re-journal a value
            # that just came off disk, and the restore is not a user action.
            self._goal_state.set(stored.goal)

        # ``(what was missed, which command re-attaches it)``, so the notice can
        # name the recovery step per slot the way every sibling miss-notice in
        # the TUI does ("Run /team to list teams, …").
        missing: list[tuple[str, str]] = []
        # True only when a lookup ran to completion and answered "no such
        # thing". A registry that is absent or that RAISED has established
        # nothing about whether the team still exists, and saying "renamed or
        # deleted" for those sends the operator off to re-create something that
        # is still there (D2/R3). Both of those are also the transient cases the
        # carried-name recovery exists for.
        looked_up_and_absent = True
        if stored.team:
            team = None
            registry = self.team_registry
            if registry is None:
                looked_up_and_absent = False
            else:
                try:
                    team = registry.get_team_by_name(stored.team)
                except Exception:  # noqa: BLE001 — a bad registry row is not fatal
                    team = None
                    looked_up_and_absent = False
            if team is None:
                missing.append((f"team {stored.team!r}", "/team"))
                # Carried so the next mutation cannot erase it (R1).
                self._unresolved_team = stored.team
            else:
                # Through ``attach_team`` rather than by setting the brief, so
                # ``active_team`` (what subagents inherit) is restored too, not
                # just the prompt text.
                self.attach_team(team)
        if stored.agent:
            resolved = None
            try:
                resolved = self.attach_agent_profile(stored.agent)
            except Exception:  # noqa: BLE001 — same contract as the team path
                resolved = None
                looked_up_and_absent = False
            if resolved is None:
                missing.append((f"agent {stored.agent!r}", "/agent"))
                self._unresolved_agent = stored.agent
        if missing:
            #: Held rather than logged, because the person who needs to know is
            #: the one looking at a band segment that did NOT come back. The TUI
            #: reads this once on adopt and says so in its ordinary
            #: system-notice style; headless hosts may ignore it.
            #
            # Names what did NOT come back, never a blanket "unattached": a
            # partial miss is the common case (a deleted team beside a profile
            # that resolved fine), and claiming the whole session is bare would
            # contradict the band segment still showing the half that restored.
            #
            # Copy shape follows the established miss-notice idiom in the TUI:
            # ONE em-dash clause separator (never a parenthetical plus a
            # semicolon, which was a third idiom — D3), and it ends on the
            # RECOVERY rather than on the damage, because "run /team to
            # re-attach" is the only part the user can act on (D2).
            what = " and ".join(name for name, _ in missing)
            # Deduped: a both-missing restore would otherwise say "/team or
            # /team". ``dict.fromkeys`` is the file's ordered-set idiom.
            commands = " or ".join(dict.fromkeys(command for _, command in missing))
            # "from the previous session" is dropped deliberately: this notice
            # only ever fires on adopt after a resume, "restore" already says
            # it, and the words cost 26 cells that pushed the common one-slot
            # case to 101 characters — over a 100-column terminal, so it wrapped
            # to two lines for no added meaning (D3).
            #
            # The cause clause is dropped as well once BOTH slots are missing:
            # naming two things already costs ~35 cells, and keeping the clause
            # there put the line back over 100 and wrapped it again. Of the two,
            # the cause is the part the user cannot act on — WHAT is gone and
            # HOW to get it back are both load-bearing — so it is the one that
            # yields. A both-missing restore is also overwhelmingly a moved
            # config directory rather than two independent deletions, which is
            # the case the clause would describe least accurately anyway.
            cause = " — renamed or deleted" if looked_up_and_absent and len(missing) == 1 else ""
            self.attachment_restore_notice = (
                f"could not restore {what}{cause}. Run {commands} to re-attach."
            )

    def _resolve_profile_or_specialist(self, name: str) -> tuple[str | None, Any, str, str]:
        """Resolve a NAME to an attachable profile, priority order fixed here.

        The SINGLE source of truth for how ``/agent`` attach and a team's
        manager resolution turn a name into a persona, so the two can never
        disagree about which of a role, a specialist, and a packaged seed wins
        (the A1 bug and its team twin were exactly that disagreement).

        Order, strongest first:

        1. the operator's own registered ROLE — ``resolve_profile`` returns a
           profile with a non-``None`` ``agent_id`` only for a real registry
           role (never a packaged seed), so an ``agent_id`` here is the
           operator's own role and outranks everything below;
        2. the operator's own SPECIALIST — checked BEFORE the seed fallthrough,
           which is the whole fix: ``resolve_profile`` honours only role rows
           and otherwise returns the SEED, so a specialist named after a seed
           word would otherwise be shadowed by that seed;
        3. a packaged SEED resolved by ``resolve_profile`` (``agent_id`` is
           ``None``), so ``reviewer`` and friends still resolve on a fresh
           machine with no registry row of that name.

        Returns ``(kind, profile, specialist_prompt, display_name)`` where
        ``kind`` is ``"role"``/``"seed"`` (``profile`` set, ``specialist_prompt``
        empty), ``"specialist"`` (``profile`` ``None``, ``specialist_prompt``
        set), or ``None`` (nothing attachable by that name — ``profile`` ``None``
        and both strings empty). Ordinary conversational/autosave rows are not
        attachable: only an explicit ``is_specialist`` marker or a role tag
        qualifies, so a private chat agent's prompt is never pulled in by a
        coincidental name.
        """
        # Delegates to the ONE shared resolver in ``agent_profiles`` so this
        # session path, ``/agent`` attach, and the org-chart resolver cannot
        # disagree about which of a role/specialist/seed a name picks (the A1
        # bug was that disagreement). This method stays as the session's named
        # entry point — callers here and the tests that pin the order reference
        # it — but the logic lives in one place now.
        from local_operator.agent_profiles import resolve_profile_or_specialist

        return resolve_profile_or_specialist(name, registry=self.agent_registry)

    def attach_agent_profile(self, name: str) -> str | None:
        """Attach a named role/specialist's instructions to THIS session.

        The TUI's ``/agent <name>`` surface. Distinct from launching the
        profile as a subagent (``task(agent=...)``): here the user chooses to
        make the CURRENT conversation speak with that profile's instruction
        set, no child process or fresh session involved.

        The instructions ride the volatile tail (see ``build_system_blocks``)
        exactly like :meth:`attach_team`'s brief, so an attach mid-session
        never invalidates the cached persona prefix. Interaction with a team
        is deliberate: the two briefs live in SEPARATE fields and coexist — a
        ``/team`` manager can adopt a specialist's voice without dropping the
        roster — while a later ``/agent`` replaces only the earlier agent
        brief, because the user is switching hats, not stacking them.

        Resolution ORDER matters and must match ``_agent_profile_rows`` in the
        TUI, or listing and attach disagree. It is delegated to the ONE shared
        resolver ``_resolve_profile_or_specialist`` (own role, then own
        specialist BEFORE a same-named packaged seed, then the seed) so this
        path and the team-manager path cannot drift — see that method for why
        the specialist has to be checked before the seed (the A1 bug).

        Returns the resolved profile's display name, or ``None`` when nothing
        by that name is a role or specialist (the caller reports it; a typo
        must not half-attach anything).
        """
        kind, profile, specialist_prompt, display_name = self._resolve_profile_or_specialist(name)
        if kind in ("role", "seed") and profile is not None:
            return self._stamp_agent_brief(profile.preamble.strip(), profile.name)
        if kind == "specialist":
            # Tagged with the specialist's name so the model can tell whose
            # voice this is — the same shape a role preamble carries.
            body = f"[agent: {display_name}]\n{specialist_prompt}" if specialist_prompt else ""
            return self._stamp_agent_brief(body, display_name)
        return None

    def clear_agent_profile(self) -> None:
        """Detach any active ``/agent`` profile (the ``/agent clear`` verb).

        Mirrors ``set_goal("")`` clearing the goal: the brief rides the
        volatile tail, so blanking it drops the persona from the next turn
        without touching the cached prefix or the separately-held team brief.
        Idempotent — clearing when nothing is attached is a no-op the caller
        can still report plainly.
        """
        self._goal_state.agent_brief = ""
        # Blank the NAME as well, not just the brief. The band's active-profile
        # segment (U2) reads ``active_agent`` (i.e. ``agent_name``), so a detach
        # that dropped only the brief would leave ``◉ auditor`` painted next to a
        # "no agent active" notice — the two surfaces of the same detach
        # contradicting each other. Both fields are stamped together in
        # ``_stamp_agent_brief``; they must be cleared together too.
        self._goal_state.agent_name = ""
        # An explicit detach also retires a carried unresolved name (R1), or
        # ``/agent clear`` would appear to work and the profile would come back
        # at the next resume.
        self._clear_unresolved("agent")
        # A detach is as much a fact to survive a resume as an attach: without
        # this the sidecar would still name the profile the user just dropped,
        # and the next resume would silently re-attach it.
        self._persist_attachment()
        self.refresh_frontend_state()

    def _stamp_agent_brief(self, body: str, display_name: str) -> str:
        """Store the resolved brief on the volatile tail and report success.

        A resolved NAME with empty instructions (a role or specialist that
        says nothing) still counts as attached: the persona layered nothing,
        but reporting failure would send the user hunting for a typo that does
        not exist. See A2 — the caller distinguishes the empty-brief case in
        its notice rather than rejecting it here.
        """
        self._goal_state.agent_brief = body
        # The NAME is stamped even when ``body`` is empty (the A2 hollow-profile
        # case): the profile IS attached and the band must name it, so the
        # segment tracks "which profile is in force", not "did it layer text".
        self._goal_state.agent_name = display_name
        # A successful attach supersedes any carried unresolved name (R1).
        self._clear_unresolved("agent")
        # The single funnel every successful ``/agent`` attach passes through
        # (role, seed and specialist all land here), so journalling once from
        # this point cannot miss a path the way three call-site writes could.
        self._persist_attachment()
        self.refresh_frontend_state()
        return display_name

    @property
    def variables(self) -> Any:
        """The session's variable store, including memory-only credentials.

        Exposed so a front end can store, list and forget session credentials
        without reaching into ``_variables``. ``None`` on a session that was
        built without a store (embedded SDK callers, some test doubles).
        """
        return self._variables

    @property
    def conversation_name(self) -> str:
        """The conversation's title ("" until one is set or generated)."""
        return self._conversation_name.text

    def _display_session_name(self) -> str:
        """The conversation title DISPLAY surfaces should show for this session.

        Identical to :attr:`conversation_name` for a top-level session. On a
        SUBAGENT — which has no title of its own and can never generate one,
        since naming runs in the TUI host and the owned-session runtime and a
        one-shot child passes through neither — it resolves to the PARENT's
        live title instead, so a delegated tab group reads
        ``<parent conversation> › <job label>`` rather than a bare fallback.

        Resolved on every call, never cached: the parent is normally named a
        second or two into its first turn while its children are launched
        later, so a value snapshotted at the child's construction would be the
        empty string for the child's whole life. That staleness is the exact
        failure this method exists to close, and it is why
        :meth:`_build_tool_context` passes this as the PROVIDER as well as the
        snapshot.

        RECURSIVE by construction, because delegation nests: a child of a
        top-level session keeps ``task``/``wait``/``jobs`` (see
        ``harness.subagent``), so a manager fanning out to workers is depth 2
        and the operator's usual shape. The middle child has no title of its
        own and can never grow one, so asking it for its *holder* yields ""
        forever and a grandchild fell back to the cwd every sibling of every
        conversation shares — two ``qa`` grandchildren under two different
        conversations rendered identically, the very collision this method
        exists to prevent. Asking the parent for its resolved DISPLAY name
        instead walks the chain to whichever ancestor actually holds a title.
        The walk is bounded by the lineage depth built at construction (no
        cycle is constructible: a parent always predates its child).

        Display only. Identity and authorization stay on ``session_id`` — a
        child borrowing its parent's name for a tab pill must never be read as
        a child acting with its parent's identity.

        Deliberately does NOT reach for a parent's title on an unnamed FORK:
        a fork is not a child, holds no ``_parent_display_name``, and
        ``_load_conversation_name`` declines the inheritance on purpose (see
        :attr:`wears_inherited_title`). Its display label stays the cwd
        substitution every unnamed session gets.
        """
        own = self._conversation_name.text
        if own or self._parent_display_name is None:
            return own
        return self._parent_display_name()

    @property
    def wears_inherited_title(self) -> bool:
        """True while this session is a FORK that has not yet named itself.

        The picker's row marker answers this question from disk
        (``resume.wears_inherited_title``); this is the same fact for the
        session that is RUNNING, so a host can tag the surfaces that name a live
        conversation — the terminal/tab title above all.

        It matters most exactly where it is least visible. An unnamed fork's
        ``conversation_name`` is EMPTY by design (:meth:`_load_conversation_name`
        declines the parent's title), so every host falls back to a label
        derived from the replayed history — which is the parent's opening
        message — or to the working directory. Both make a fork's window read
        identically to its parent's in a switcher, which is the one surface a
        user scans to find the window that just opened.

        **Costs no I/O.** ``_is_unnamed_fork()`` already runs once at
        construction inside :meth:`_load_conversation_name` and its verdict was
        thrown away; this keeps it. The flag is cleared in
        :meth:`set_conversation_name` — the one path a fork's own name can
        arrive by — so the tag marks the ambiguous STATE and not ancestry,
        exactly as the picker's marker does.
        """
        return self._wears_inherited_title

    @property
    def conversation_name_state(self) -> ConversationName:
        """The title holder itself — hosts need the ``user_set`` precedence
        flag and the once-only request latch, not just the string."""
        return self._conversation_name

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        """Name the conversation; returns the title in force afterwards.

        ``user_set=True`` (an explicit rename) wins permanently: a generated
        title landing later is discarded rather than overwriting a name the
        user chose. Auto-naming passes ``user_set=False``.

        Journalled as a side effect, so the name survives to the next
        ``--resume``. The write is FIRE-AND-FORGET on purpose: this method is
        called from the TUI's synchronous paint path (``_store_title``,
        ``_cmd_rename``), which cannot await, and a title is decoration — a
        transcript append that fails must cost the name on disk, never the
        turn. Only a store that actually CHANGED something is journalled: a
        generated title that lost to a user-set one returns the standing name,
        and re-appending it would grow the transcript by a line per turn.
        """
        before = (self._conversation_name.text, self._conversation_name.user_set)
        stored = self._conversation_name.set(text, user_set=user_set)
        if (stored, self._conversation_name.user_set) != before:
            # The fork now has a name of its own, so it is no longer wearing an
            # inherited one and every surface must stop saying that it is. This
            # is the ONLY way a name reaches a live session (both the naming
            # errand and `/rename` land here), so clearing it here cannot be
            # forgotten by a third caller.
            self._wears_inherited_title = False
            self._conversation_name_dirty = True
            self._spawn_conversation_name_write()
            # Mirror the human name into the analytics ledger so the
            # ``/analytics /usage`` per-session table shows a title rather than
            # a 12-hex id. Best-effort and off the hot path (the recorder runs
            # the upsert on its own thread); a failure here must never cost the
            # rename, which is why it is guarded and never awaited.
            if stored:
                try:
                    from local_operator.analytics import get_recorder

                    get_recorder().note_session_name(self.session_id, stored)
                except Exception:  # noqa: BLE001 — analytics is best-effort
                    logger.debug("analytics: note_session_name failed", exc_info=True)
            # Rename the browser tab group this session already opened. The
            # title normally lands a second or two INTO the first turn, which is
            # after an opening "look at this page" created the group, so without
            # a push the group keeps the label the session had at open time.
            # Every ordinary browser command reconciles the group as a side
            # effect, but open -> screenshot -> close issues none, so that
            # session never self-heals. Fire-and-forget for the same reason the
            # journal write is: this runs on the TUI's synchronous paint path
            # (``_store_title``, ``_cmd_rename``), which cannot await, and tab
            # chrome must never delay or fail a rename.
            self._push_browser_title()
        self.refresh_frontend_state()
        return stored

    def _push_browser_title(self) -> None:
        """Best-effort rename of the open browser tab group; never raises.

        Skipped entirely when this session has no tab open, which is the common
        case: a session that never browses issues no RPC at all. The import is
        function-local to match ``_close_browser_surface`` next door rather than
        to save anything — ``tools.builtin`` is already imported at module scope
        here (see the top of this file), so unlike that sibling's claim there is
        no import cost to defer (QA round 1).
        """
        if not self._browser.surface_id:
            return

        async def _push() -> None:
            try:
                from local_operator.tools.builtin import retitle_browser_surface

                await retitle_browser_surface(self._browser, self._build_tool_context())
            except Exception:  # noqa: BLE001 — tab chrome is never worth a failure
                logger.debug("could not push the session title to the browser", exc_info=True)

        # Through the tracked task group, so disposal cancels it: unlike the
        # title's journal write, a rename that misses because the session is
        # closing costs nothing — the tab is going away with it.
        try:
            self._spawn_background(_push())
        except RuntimeError:
            # No running loop (a session constructed and named outside one) —
            # the same case ``_spawn_conversation_name_write`` guards one method
            # below, and for the same reason: ``_spawn_background`` falls back
            # to ``ensure_future``, which raises without a loop. The guard has
            # to sit at the SCHEDULING call, because the swallow inside
            # ``_push`` is inside the coroutine and this raises before it ever
            # runs. A rename is decoration; ``set_conversation_name`` runs on
            # the TUI's synchronous paint path and must return the stored title
            # rather than take its caller down with it. ``_spawn_background``
            # closes both coroutines before re-raising, so nothing leaks.
            logger.debug("no running loop; skipped the browser tab-group rename")

    def _spawn_conversation_name_write(self) -> None:
        """Start (or coalesce onto) the background journal write for the title.

        Deliberately NOT ``_spawn_background``. That helper's tasks are
        cancelled wholesale by ``dispose``, and a name stored in the closing
        moments of a session is exactly the case that must still reach disk —
        so this task is tracked separately and AWAITED by
        :meth:`_flush_conversation_name` instead of being cancelled with the
        rest.

        One task at a time, because the payload is read at write time rather
        than captured at call time: a rename landing while a write is in flight
        is already covered by that write's own read, and a second task would
        append a duplicate row for one change.
        """
        if self._disposed:
            return
        task = self._conversation_name_task
        if task is not None and not task.done():
            return
        try:
            task = asyncio.ensure_future(self._persist_conversation_name())
            # Consume the exception explicitly. The failure is already intended
            # to be swallowed (a title must never cost a turn), but nothing
            # retrieves the result of a task that finishes BEFORE the dispose
            # flush looks at it, and asyncio then reports "Task exception was
            # never retrieved" on the loop's error handler at GC time — log
            # noise blaming the session for a disk error it deliberately
            # tolerated. Logged at debug because the dispose flush retries.
            task.add_done_callback(self._on_conversation_name_written)
            self._conversation_name_task = task
        except RuntimeError:
            # No running loop (a session constructed and named outside one).
            # The dispose flush is the backstop; the name is not lost.
            self._conversation_name_task = None

    @staticmethod
    def _on_conversation_name_written(task: "asyncio.Future[None]") -> None:
        """Retrieve the title write's outcome so asyncio does not report it.

        The write is fire-and-forget by design, so a failure here is expected
        to be tolerated rather than raised — but an exception nobody reads is
        reported by the loop at collection time, which blames the session for a
        failure it chose to survive. Reading it is the whole job; the dispose
        flush is what actually retries.
        """
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.debug("conversation name write failed; will retry at dispose", exc_info=error)

    def _is_unnamed_fork(self) -> bool:
        """True while this session is a fork still wearing its parent's title.

        A fork must be named for what it was forked to DO, not for what the
        parent was doing — so the inherited title is declined until the fork has
        one of its own (see :meth:`_load_conversation_name`).

        "Of its own" is decided by TIME, not by a flag, because the parent's
        title and the fork's own title are the same kind of entry in the same
        copied file and nothing else distinguishes them. Any title journalled
        AFTER the moment of the fork was written by this session; anything at or
        before it came across in the copy. That makes the suppression one-shot
        without any state to reset: the first name this fork writes lands after
        ``forked_at``, and from then on every resume restores it normally.

        Reads the origin marker, not a cached field, because this runs during
        construction — before any of the session's own state exists — and is
        asked once per boot. Tolerant of a missing or unreadable marker in the
        direction that preserves existing behaviour: no marker means "not a
        fork", so an ordinary session's restore is untouched.
        """
        try:
            from local_operator.fork import fork_instant

            forked_at = fork_instant(self._transcript.directory)
        except Exception:
            return False
        if forked_at is None:
            return False
        entry = self._transcript.latest_custom_entry(CONVERSATION_NAME_CUSTOM_TYPE)
        if entry is None:
            # A fork of a conversation that was never named. Nothing to decline,
            # and the naming path fires on its first turn like any new session.
            return True
        # ``>`` and not ``>=``: an entry written in the same instant as the fork
        # is the parent's, since the clone necessarily happened after it.
        return not entry.ts > forked_at

    def _load_conversation_name(self) -> None:
        """Adopt the title journalled by the session this one resumes.

        Silently tolerant of a malformed entry, matching
        :meth:`_load_wake_schedules`: a resume must not be refused because a
        decoration could not be read. ``user_set`` rides along because it is
        precedence, not display — a name the USER typed has to keep outranking
        generated titles across a resume, or the first re-title check in the
        resumed session would quietly overwrite it.

        Writes through the holder's fields rather than through
        :meth:`ConversationName.set`, since ``set`` cannot express "restore a
        user-set title" without also claiming it as a fresh user action, and a
        restore is neither a rename nor a generation — it is the same name it
        already was.
        """
        # A FORK THAT HAS NOT YET BEEN NAMED DELIBERATELY DECLINES THE
        # INHERITANCE. A fork carries a byte-identical copy of its parent's
        # transcript, so the parent's journalled title is sitting right there in
        # it — and adopting it would name the branch after the work it left,
        # then latch ``requested`` and never reconsider. The user forked to do
        # something ELSE, and the name they will scan the picker for is the name
        # of that something.
        #
        # Suppressed HERE, at adoption, rather than by editing the copied
        # transcript or withholding the title sidecar. Both alternatives were
        # considered and are worse:
        #
        # - Rewriting the clone to strip the entry would destroy the
        #   byte-identity the fork's cache warmth depends on (the fork's first
        #   request has to reproduce the parent's cached prefix exactly).
        # - Not copying ``title.json`` would not work at all: the title arrives
        #   through this transcript entry regardless, and the sidecar's real job
        #   is to keep the fork's PICKER ROW labelled during the seconds before
        #   its own name lands — without it a brand-new fork renders as a blank
        #   row, since its opening message is the parent's.
        #
        # The suppression is one-shot by construction: it applies only while the
        # fork has no title of its own, so once the naming path writes one, every
        # later ``/resume`` of this fork restores that name normally.
        if self._is_unnamed_fork():
            # Remembered rather than recomputed: this is the one place the
            # question is asked (it needs the origin marker and the transcript's
            # newest title entry, both of which are in hand here at boot and
            # neither of which a host should re-read per frame), and the answer
            # is what every surface naming this session needs in order not to
            # display it under the parent's identity. See
            # :attr:`wears_inherited_title`.
            self._wears_inherited_title = True
            return
        details = self._transcript.latest_custom(CONVERSATION_NAME_CUSTOM_TYPE)
        if not details:
            return
        text = details.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        self._conversation_name.text = " ".join(text.split())[:MAX_TITLE_CHARS]
        self._conversation_name.user_set = bool(details.get("user_set"))
        # A restored conversation has already SPENT its naming attempt: the
        # title on disk is the result of it. Without this latch a host that
        # asks "has naming been requested?" would spend a second provider call
        # to re-derive a name the session is already wearing.
        self._conversation_name.requested = True

    async def _persist_conversation_name(self) -> None:
        """Journal the title in force (newest entry wins on replay).

        The payload is SNAPSHOTTED before the append and compared against the
        holder afterwards, and the dirty flag is cleared only when they still
        agree. Two races make that necessary rather than fussy:

        * A rename landing while the append is in flight (the file lock is
          held, so the window is real) leaves a newer title in the holder than
          the row that just went to disk. Clearing unconditionally marked that
          newer title as saved and the next ``--resume`` restored the OLD one —
          reproduced with a transcript whose append was slowed to 150 ms.
        * A write cancelled at teardown never reaches this line at all, so the
          entry still reads as outstanding to :meth:`_flush_conversation_name`
          and is retried there instead of lost.

        Left dirty, the write is re-driven by the flush at dispose, so the last
        title a user chose is the one on disk.
        """
        # A LOOP with a hard cap, not recursion. The chase below is bounded in
        # practice by renames actually landing inside a write, but that is a
        # behavioural assumption about how fast a human types rather than a
        # structural one — driven by a title that always moves, the recursive
        # form raised ``RecursionError`` after ~3000 appends. A cap makes the
        # bound structural, and the cost of hitting it is one stale title on
        # disk that the dispose flush will still correct.
        for _ in range(_NAME_PERSIST_MAX_PASSES):
            payload = {
                "text": self._conversation_name.text,
                "user_set": self._conversation_name.user_set,
            }
            await self._transcript.append_custom(CONVERSATION_NAME_CUSTOM_TYPE, payload)
            # Journal the same title to the sidecar the picker reads O(1), on
            # the SAME event that writes it to the transcript. This is what
            # makes a title in the untouched middle of a large transcript
            # findable without a full read on the picker's synchronous path —
            # see resume.write_session_title / TITLE_SCAN_BYTES. Imported here
            # rather than at module top so the CLI's import-guard on
            # ``resume`` is unaffected, and best-effort by the helper's own
            # contract: a failed sidecar write never fails a turn.
            from local_operator.resume import read_title_names, write_session_title

            session_dir = self._transcript.directory
            write_session_title(
                session_dir,
                str(payload["text"]),
                user_set=bool(payload["user_set"]),
                past_names=read_title_names(session_dir),
            )
            if (self._conversation_name.text, self._conversation_name.user_set) == (
                payload["text"],
                payload["user_set"],
            ):
                self._conversation_name_dirty = False
                return
            # The title moved under the append. Chase it now rather than leaving
            # the newer name to the dispose flush: a session can run for hours
            # after a rename, and "correct only if you quit" is not a property
            # worth having when the retry is one more append. Each pass narrows
            # the window.

    async def _flush_conversation_name(self) -> None:
        """Land an outstanding title write before the session tears down.

        The ordinary write is a background task and :meth:`dispose` CANCELS
        background tasks, so a title stored in the closing moments of a session
        — a ``/rename`` just before ctrl+d, or a generated title arriving as
        the user quits — would be cancelled before reaching disk and the next
        ``--resume`` would open nameless. Reproduced exactly that way: a name
        set and then disposed with no intervening await never got a turn of the
        event loop, and the transcript carried no entry at all.

        Any write already in flight is awaited first, so the ordinary path
        costs nothing here and the retry below only runs when that write never
        happened (never scheduled, or cancelled).

        Both halves share ONE deadline, and it is the whole cost of this method
        rather than the cost of each half. Charged separately they ran in
        sequence, so teardown could take twice the advertised bound; and because
        the bound decides whether a slow-but-real write is LOST rather than
        merely un-waited-for, two tight budgets were also the wrong shape — a
        4.9 s append lost its title where a single 5 s budget keeps it. An
        append is sub-millisecond on a local disk, so seconds already mean a
        stalled mount, and in that state a user is better served by a few
        seconds of teardown than by a resume that opens unnamed.
        """
        deadline = time.monotonic() + _NAME_FLUSH_TIMEOUT_S

        task = self._conversation_name_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=max(0.0, deadline - time.monotonic())
                )
            except BaseException:  # noqa: BLE001 — fall through to the retry
                pass
        if not self._conversation_name_dirty:
            return
        try:
            # BOUNDED, like the shielded wait above it. ``_persist_conversation_name``
            # takes the transcript lock, and dispose's turn-abort wait is itself
            # shielded — so a turn that outruns its 5 s budget keeps running and
            # keeps holding that lock. An unbounded await here made dispose hang
            # behind it indefinitely (measured blocking >3 s and still going),
            # which trades a missing title for a session that will not close.
            # Every other await on this path is bounded; this one has no claim
            # to be the exception. What remains of the shared deadline is the
            # budget, so the two waits cannot add up.
            await asyncio.wait_for(
                self._persist_conversation_name(),
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except Exception:
            # Decoration must never be the reason a dispose fails: losing the
            # name is survivable, losing the conversation is not. A timeout is
            # one of these — ``TimeoutError`` is an ``Exception`` — so a lock
            # held past the budget costs the title and nothing more.
            logger.warning("failed to persist the conversation name", exc_info=True)

    @property
    def wake_scheduler(self) -> WakeScheduler:
        """Exposed so the wake tool can list/create/cancel schedules."""
        return self._wake

    async def preflight_usage(self, *, consume_boundary: bool = True) -> None:
        """Run the stream's message-boundary quota check without starting a turn.

        The TUI calls this after subscribing, so an exhausted default provider
        becomes a visible warning while startup itself remains successful.

        ``consume_boundary=False`` is the ``/model`` switch probe: it evaluates
        the NEW selector immediately (a spent family cap on the previous model
        must not ride across the switch) without spending the message-boundary
        token the next request's effort classification is owed.
        """
        preflight = getattr(self._stream_fn, "preflight_usage", None)
        if not callable(preflight):
            return
        # The kwarg rides only when set: stream fakes that predate the
        # switch probe declare the bare ``(model)`` shape.
        result = (
            preflight(self._model)
            if consume_boundary
            else preflight(self._model, consume_boundary=False)
        )
        if inspect.isawaitable(result):
            await result

    # -- driving turns --------------------------------------------------------
    async def prompt(
        self,
        text: str,
        images: Sequence[ImageContent] | None = None,
        *,
        message_id: str | None = None,
        producer_command_id: str | None = None,
        admitted: asyncio.Future[None] | None = None,
    ) -> None:
        """Run one user turn to completion (awaitable) or raise.

        ``producer_command_id`` and ``admitted`` form the continuation
        admission seam: the caller receives a receipt only after the explicitly
        marked user row is on disk, while the model turn may continue afterward.
        ``message_id`` remains independent conversation identity; ordinary local
        prompts omit producer provenance even if a host supplies a message id.

        ``images`` are attachments the user pasted into their prompt; they
        ride the same message as the text so the model sees them as one
        turn rather than as a separate observation.

        Reentrancy: ``_turn_lock`` is consulted FIRST — if a live turn (user
        prompt or wake delivery) holds it, a concurrent ``prompt`` is
        rejected outright instead of queueing behind it — including the lock an
        on-demand compaction holds, which the rejection names. ``_is_streaming``
        is then re-checked under the lock to close the race where streaming was
        set between the lock probe and the acquire.
        """
        if self._disposed:
            raise RuntimeError("session is disposed")
        if self._turn_lock.locked():
            # An on-demand compaction holds the same lock a turn does, and for
            # the same reason — it is rewriting the history a request would be
            # built from. Saying "already streaming" for it would send the user
            # looking for a turn that is not there.
            raise RuntimeError(
                "context compaction is running; the prompt can be sent once it finishes"
                if self._compacting
                else "session is already streaming; use steer() to inject mid-turn"
            )
        await self._turn_lock.acquire()
        try:
            # Close the narrow completion-after-final-flush race: a shell
            # receipt may have queued while the previous owner still held the
            # lock. It must be visible before this prompt builds its request.
            await self._flush_shell_records()
            # Same race, same window: a journal notice parked by the previous
            # owner must reach the model before this prompt's request is built,
            # or the switch it announces goes unmentioned for another turn.
            self._flush_context_journal()
            # A fresh user prompt supersedes any earlier interrupt request.
            self._abort_requested = False
            if self._is_streaming:
                raise RuntimeError("session is already streaming; use steer() to inject mid-turn")
            # INLINE a pending wake catch-up ahead of the user's message, in
            # the SAME turn: the missed wakes belong before the work they were
            # meant to start, and spawning the catch-up as a competing
            # ``_prompt_messages`` would queue it behind this one (round 3,
            # M2). One turn, user message LAST, so the model reads the folded
            # wakes first and the reply still answers the user.
            catchup = self._take_resume_catchup()
            if catchup is not None:
                # Folded ahead of a real user prompt, the catch-up precedes
                # the work it shares the turn with — the same busy-path
                # obligation applies, so the text carries the guidance here
                # too. On a FRESH session this prompt starts the work rather
                # than resuming it, so the note names that instead of
                # claiming work was already under way.
                fresh = not self._context.messages
                catchup.details["text"] = self._append_busy_resume_note(
                    str(catchup.details["text"]),
                    continue_what=(
                        "continue with the user's request that follows" if fresh else None
                    ),
                )
            user = Message.user(text, images, **({"id": message_id} if message_id else {}))
            initial: list[AgentMessage] = [catchup, user] if catchup is not None else [user]
            await self._run_turn_pipeline(
                initial,
                admitted=admitted,
                admitted_id=user.id,
                producer_command_id=producer_command_id,
            )
        finally:
            self._turn_lock.release()

    async def seed_history(self, messages: list[Message]) -> None:
        """Prime the conversation from a host-supplied history.

        The server facade uses this for the two paths where the transcript is
        not already the history source: stateless ``/v1/chat`` (the caller's
        ``context`` array) and agent chat with ``persist_conversation=false``
        (the registry's stored conversation). Without it the provider sees
        the bare prompt while the response envelope echoes history the model
        never read.

        Once-only and pre-prompt: a no-op when the context already carries
        messages (transcript replay populated them) or after the first turn.
        Messages are persisted as replayed entries so later turns, pruning and
        compaction treat them like any other history.
        """
        if self._seeded or self._context.messages or self._is_streaming:
            return
        self._seeded = True
        for message in messages:
            await self._transcript.append_message(message)
            self._context.messages.append(message)

    def steer(
        self,
        text: str,
        images: Sequence[ImageContent] | None = None,
        *,
        message_id: str | None = None,
        producer_command_id: str | None = None,
    ) -> None:
        """Inject an identified steering message into the running turn.

        Attachments ride along for the same reason they do on ``prompt``. The
        optional producer identity is persisted when the queue drains, letting
        reconnecting followers deduplicate the correction like an ordinary turn.
        """
        message = (
            Message.user(text, images, id=message_id) if message_id else Message.user(text, images)
        )
        self._steering_queue.put_nowait(message)
        if producer_command_id is not None:
            self._steering_producers[id(message)] = producer_command_id
        self.refresh_frontend_state()

    def queued_steering(self) -> list[AgentMessage]:
        """A FIFO snapshot of the steering queue, without draining it.

        The read half of the recall seam: a host deciding what is still
        recallable (`SessionProtocol.recall_steering`) must be able to SEE
        the queue, and the queue itself is engine-internal. Identity of the
        entries is preserved, which is what hosts match on.

        ``asyncio.Queue`` has no public peek, so the snapshot is a drain and
        a rebuild through the get/put API — the same shape
        :meth:`recall_steering` uses, minus the removal. A drain CAN
        interleave with this rebuild — ``_drain_steering`` awaits a disk
        append between its ``get_nowait``s, and a key handler runs on the
        same loop — and the interleaving is BENIGN: a message the drain
        already holds simply fails the caller's identity check (it is no
        longer in the queue, so it is not recallable), and the re-put items
        join the same boundary's delivery. What this method promises is only
        that it never loses or reorders an entry, which the get/put round
        trip keeps.
        """
        snapshot: list[AgentMessage] = []
        while not self._steering_queue.empty():
            snapshot.append(self._steering_queue.get_nowait())
        for message in snapshot:
            self._steering_queue.put_nowait(message)
        return snapshot

    def steer_message(self, message: Message) -> None:
        """Queue a caller-built steering message, sharing the caller's object.

        The identity-preserving twin of :meth:`steer`: a host that keeps its
        own reference to the queued message (the TUI pairs it with the
        transcript blocks it painted for it) can later ask for THAT message
        back via :meth:`recall_steering`. ``steer`` builds the Message here,
        so no reference the host could match on ever leaves this method.
        """
        self._steering_queue.put_nowait(message)
        self.refresh_frontend_state()

    def request_fork(
        self,
        config_dir: Path,
        *,
        message: str = "",
        on_complete: Callable[[str, str], None],
    ) -> bool:
        """Branch this conversation at the next safe boundary. Replacement? -> True.

        The transcript on disk is the artifact a fork copies, so a fork is only
        correct at a point where the run's messages are already persisted AND
        the file does not end in an assistant ``tool_use`` with no
        ``tool_result`` — a clone taken mid-batch would make the fork's very
        first request a provider 400, in a different window, minutes later.

        Two drain points, and BOTH are required:

        * :meth:`_on_turn_end`, the tool-loop boundary, which is where a fork
          requested during a multi-step run lands. That hook opens with
          ``_persist_progress`` unconditionally, so the run's history is on disk
          by the time the clone runs — exactly the precondition a file copy
          needs.
        * the post-run path after ``_persist_new_messages``, for a fork
          requested during the FINAL model turn. ``on_turn_end`` fires only when
          the loop will CONTINUE (``harness/loop.py`` guards it on
          ``has_more_tool_calls``, because a terminal boundary is the post-run
          pass's job), so a fork asked for near the end of a run never reaches
          it. Implementing only the hook leaves that fork hanging until the
          user's next turn, which presents as "sometimes /fork does nothing".

        When no turn is running the caller should clone directly instead of
        coming here: the transcript is already complete and consistent at idle,
        so deferring would only add latency to the common case.

        ``on_complete`` is called with ``(fork_id, error)`` — exactly one of
        which is non-empty — on the session's own loop. The host owns everything
        the user sees from there; the session's responsibility ends at the clone.

        Returns True when this request REPLACED one that was already pending, so
        the host can say so rather than silently dropping the earlier one.
        """
        # A second request REPLACES the first, and says so. A bare assignment
        # discarded the earlier one silently: both `/fork`s were echoed as user
        # rows and both got "forking at the next safe boundary…", so the screen
        # showed two forks pending while exactly one was ever created — carrying
        # only the second message. A user refining their instruction got what
        # they wanted by luck; a user deliberately branching twice from the same
        # point lost a branch, and the single receipt arrives minutes later and
        # far up the scrollback from the two acknowledgements, so the
        # discrepancy is very unlikely to be noticed.
        #
        # Replacing rather than queueing is the right default: the common case
        # by far is changing your mind about the message, and two windows opened
        # from one keystroke pair is the more surprising outcome. The caller
        # reports the replacement; the session only tells it that one happened.
        replaced = self._fork_pending
        self._fork_pending = _ForkRequest(
            config_dir=config_dir, message=message, on_complete=on_complete
        )
        return replaced is not None

    def cancel_fork(self) -> bool:
        """Withdraw a pending fork. True when one was actually waiting.

        Esc is what a user reaches for to take something back in this app, and
        it already unsends a queued steer (:meth:`recall_steering`). Without
        this a requested fork was the one thing in the family with no undo: the
        abort stopped the turn, which LOOKS like it stopped everything, and the
        fork then arrived minutes later as a window in a sidebar row nobody was
        watching. Wired into :meth:`abort` so the escape route the user already
        knows covers this too.
        """
        cancelled = self._fork_pending is not None
        self._fork_pending = None
        return cancelled

    def has_pending_fork(self) -> bool:
        """Whether a fork is waiting for a boundary. Polled by the loop.

        Wired to ``LoopConfig.has_pending_fork``, which ORs it into the
        interrupt poll so an ``interruptible=True`` tool is torn down within one
        poll interval (~250 ms) rather than the fork waiting out a ten-minute
        ``wait``. It deliberately does NOT reach the batch-skip branch; see that
        field's comment for why that asymmetry is the subtle part.
        """
        return self._fork_pending is not None

    async def _drain_pending_fork(self) -> None:
        """Perform a deferred fork, if one is pending. Never raises into a turn.

        Called from both boundaries :meth:`request_fork` documents. The pending
        request is cleared BEFORE the clone runs, so a failure cannot leave a
        request that re-fires at every subsequent boundary for the rest of the
        session.
        """
        request = self._fork_pending
        if request is None:
            return
        self._fork_pending = None
        from local_operator.fork import ForkError, fork_session

        try:
            # On a worker thread: the copy is small in practice (the largest
            # transcript in a real store measured 216 KB) but its size is
            # user-controlled, and a turn boundary is not a place to hold the
            # loop for unbounded file I/O.
            fork_id = await asyncio.to_thread(
                fork_session,
                request.config_dir,
                self._session_id,
                message=request.message,
            )
        except ForkError as exc:
            request.on_complete("", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — a fork must never kill a turn
            logger.exception("fork: unexpected failure cloning %s", self._session_id)
            request.on_complete("", str(exc))
            return
        request.on_complete(fork_id, "")

    def recall_steering(self, message: AgentMessage) -> bool:
        """Remove ONE specific message from the steering queue, if present.

        The TUI's Esc uses this to unsend a queued mid-turn steer: the
        message goes back to the composer and the loop's next boundary finds
        it gone. Matched by identity — the host hands back the very object
        :meth:`steer_message` queued — so an equal-but-distinct message (the
        same text steered twice) is not what a recall removes, and everything
        else in the queue keeps its place: older steers, newer steers, and
        wake deliveries, which ride this queue but were never the caller's
        object. Returns False when the message is not queued — already
        drained at a boundary, or never queued — and changes nothing.

        The rebuild goes through the public get/put API rather than the
        queue's private deque: ``asyncio.Queue`` is the contract this queue
        has with the loop, and reaching inside it would couple the session
        to an implementation detail. A concurrent ``_drain_steering`` can
        interleave between the get and the put (it awaits a disk append per
        message); the interleaving is benign for the same reasons
        :meth:`queued_steering` documents — the drain takes what it finds,
        the rebuild re-puts the rest, and nothing is lost either way.
        """
        remaining: list[AgentMessage] = []
        found = False
        while not self._steering_queue.empty():
            item = self._steering_queue.get_nowait()
            if item is message and not found:
                found = True
                self._steering_producers.pop(id(item), None)
                continue
            remaining.append(item)
        for item in remaining:
            self._steering_queue.put_nowait(item)
        if found:
            self.refresh_frontend_state()
        return found

    def _peer_custom_message(self, text: str, sender: dict[str, Any]) -> CustomMessage:
        """Build the transcript entry for one inbound cross-session message.

        ``details["text"]`` is what the MODEL reads: it is wrapped in a
        provenance envelope (mirroring the subagent-message envelope in
        ``comms.py``) so the model knows the message came from another session
        and who sent it, rather than mistaking it for the user's own turn.
        ``details["body"]`` is the raw text the UIs (TUI/phone) render, and
        ``details["sender"]`` carries the advisory identity for the indicator
        label. ``attribution="user"`` routes it through the same allow-listed
        user-turn path as a wake/hub delivery (see ``build_llm_history``).
        """
        pid = sender.get("pid")
        conversation = sender.get("conversation_name", "")
        model_label = sender.get("model_label", "")
        wrapped = (
            f"<peer-session-message "
            f"from_pid={pid!r} "
            f"conversation={conversation!r} "
            f"model={model_label!r}>\n"
            f"{text}\n"
            "</peer-session-message>"
        )
        return CustomMessage(
            custom_type=PEER_MESSAGE_MESSAGE_TYPE,
            attribution="user",
            details={"text": wrapped, "body": text, "sender": sender},
        )

    async def receive_peer_message(
        self,
        text: str,
        *,
        mode: str = "mailbox",
        wake: bool = False,
        sender: dict[str, Any] | None = None,
    ) -> str:
        """Deliver a message from ANOTHER local lop session into this one.

        This is the receive half of ``lop send``. No existing method both
        persists a message durably to the transcript AND makes it visible to
        the model without driving a turn, which is why record-only needs its
        own branch here rather than reusing ``queue_aside`` (materializes only
        at a turn boundary — a genuinely idle session has no boundary) or
        ``seed_history`` (pre-first-turn only).

        Delivery modes:
        - ``mailbox`` (default), no wake: record-only. Persist the row now so
          the human sees it immediately and a crash cannot lose it, and append
          to live context so the model reads it on its next turn. The idle
          session stays idle — non-interrupting by design.
        - ``mailbox`` + ``wake`` while idle: drive a turn now via the prompt
          pipeline (which persists the row once — do NOT also append).
        - ``steer`` while busy: put the peer row ITSELF on the steering queue
          so it is injected mid-turn (the drain persists what it takes — do
          NOT also append).
        - ``steer`` while idle: nothing to steer into, so degrade to a driven
          turn exactly like mailbox+wake idle (dropping it would violate the
          guarantee that the message MUST appear in history).

        Returns a short human-readable detail string for the sender's ack.
        """
        # Resolve the sender against the LOCAL registry before anything renders
        # or persists. The identity arrives over the wire as the sender's own
        # self-report and can be empty or pid-only (a `lop send` whose ancestry
        # walk found no session record), which painted `peer message from
        # (pid 1)` — no name, no model, nothing to follow in a busy transcript.
        # The registry is same-account, local, and written by the owning process
        # itself, so it is the authoritative answer to "who is pid N"; the
        # sender's own values win only where it actually supplied them. Done
        # HERE, at the single point where an inbound peer message enters the
        # session, so the enrichment reaches the persisted row, the live
        # receipt, AND the model-visible provenance envelope alike — putting it
        # in the registrant dispatch would leave the transcript path to
        # re-derive it, and putting it in the TUI would fix only the card.
        #
        # Imported in-function: `mobile.peer_send` reaches the registry and the
        # config path, and this module must not grow a module-level dependency
        # on the mobile package for a per-message nicety.
        from local_operator.mobile.peer_send import resolve_sender_identity

        sender = resolve_sender_identity(sender)
        message = self._peer_custom_message(text, sender)
        busy = self._is_streaming
        if mode == "steer":
            if busy:
                # Queue the peer CustomMessage ITSELF, not a plain user Message
                # built from its body (what ``steer()`` would mint). Both reach
                # the model — ``_default_convert_to_llm`` renders this custom
                # type as a user turn exactly as ``build_llm_history`` does on
                # replay — but only the CustomMessage keeps the provenance:
                # ``_drain_steering`` persists whatever it takes, so a plain
                # Message left a bare user row in the transcript (no sender,
                # no ``<peer-session-message>`` envelope, so the model read it
                # as the user's own words and a resume repainted it as one),
                # and the drain announces every plain user Message with a
                # ``MessageStartEvent`` that the TUI paints as a UserBlock
                # under the PeerMessageBlock the receipt below already painted
                # — the message showed twice. Same shape as the busy wake and
                # busy resume-catchup paths, which queue their CustomMessage.
                # The drain persists the row; appending here too would
                # double-write. Deliberately NOT counted as courtesy: a
                # ``now=True`` send is an explicit mid-turn steer, so it must
                # stay urgent for ``_has_urgent_steering`` like a typed steer.
                self._steering_queue.put_nowait(message)
                self.refresh_frontend_state()
                await self._emit_peer_receipt(message, sender)
                self._peer_arrival.mark()
                return "delivered mid-turn (steered)"
            # Idle steer has nothing to interrupt: open a turn so the message is
            # still delivered and read. _prompt_messages persists the row once.
            await self._emit_peer_receipt(message, sender)
            self._peer_arrival.mark()
            self._spawn_background(self._prompt_messages([message]))
            return "delivered (opened a turn)"
        # mode == "mailbox"
        if wake and not busy:
            # _prompt_messages persists the row through the pipeline — a
            # separate transcript/context append here would double-write.
            await self._emit_peer_receipt(message, sender)
            self._peer_arrival.mark()
            self._spawn_background(self._prompt_messages([message]))
            return "delivered and woke the session"
        # Record-only (idle without wake, or busy): persist durably NOW so the
        # human sees it and a crash cannot lose it, and make it visible to the
        # model on its next turn. The transcript write is immediate; the live
        # append routes through _append_or_park_journal, NOT a bare
        # _context.messages.append. A bare append is a splice hazard on the
        # BUSY path: appending a user-attributed message while a tool batch is
        # open leaves the live list ending assistant(tool_use) -> user with the
        # tool_results still to come, which every provider rejects and which
        # trips _pair_spliced_tool_results (same class as PR #302, C1). The
        # journal path parks the live append to the next turn-safe boundary
        # while writing the transcript now, so the record-only guarantee holds:
        # the human sees the row immediately, and the parked append lands
        # before the next turn reads context. On the idle path it appends
        # straight through, matching the prior behaviour.
        await self._transcript.append_message(message)
        self._append_or_park_journal(message)
        await self._emit_peer_receipt(message, sender)
        # LAST, deliberately: this wakes a blocking `wait`, and the woken tool
        # returns into the loop's injection boundary where _drain_steering
        # flushes the journal. Marking BEFORE the park above would let that
        # flush run before the message was parked - a lost wakeup that delays
        # delivery by a whole extra turn.
        #
        # Note what this does NOT do: it appends nothing to context. The
        # message reaches the model through the unchanged park -> flush path
        # at the post-batch boundary, which is what keeps the splice hazard
        # documented above from being reintroduced here.
        self._peer_arrival.mark()
        return "delivered to the mailbox (will be read on the next turn)"

    async def _emit_peer_receipt(self, message: CustomMessage, sender: dict[str, Any]) -> None:
        """Fire the live receipt so the owner TUI paints the indicator now.

        Mirrors ``WakeDeliveredEvent`` firing before/around the turn spawn:
        even for the record-only idle case the front end must be told the
        instant the message lands, and ``message_id`` lets it dedup this live
        receipt against the persisted row on a later history replay.
        """
        await self._emit(
            PeerMessageDeliveredEvent(
                body=str(message.details.get("body", "")),
                sender=sender,
                message_id=message.id,
            )
        )

    def set_approval_handler(self, handler: "ApprovalGate | None") -> None:
        """Install the host's tool-approval gate (see SessionProtocol).

        Read when the per-turn tool context is built rather than captured once,
        so a front end that installs its own gate after the session is already
        constructed (the TUI resolves its session in a worker, well after the
        factory ran) governs every tool call from the next one onward.

        ``ApprovalGate`` is a UNION of the two accepted shapes —
        ``(tool_name, description)`` and the same plus ``job_id`` — so a host
        that wants to know WHICH background job is asking can say so without
        every existing host having to change. Narrowing this back to the
        two-argument callable is what would silently drop the provenance at the
        last hop: the gate is installed here, so this annotation is what a host
        type-checks its own handler against.
        """
        self._request_approval = handler

    def set_ask_handler(self, handler: AskUserFn | None) -> None:
        """Install the host's interactive-question surface (see SessionProtocol).

        Read when the per-turn tool context is built, like the approval gate, so
        a front end that resolves its session in a worker still governs every
        call from the next one onward.

        Installing this is also what makes ``ask`` EXIST: the tool's createIf
        builder gates on the hook rather than on ``has_ui``, because a subagent
        inherits ``has_ui`` from its parent and has no human at its keyboard —
        a child session is built without this handler and so never advertises a
        question it could only block on.

        Which is why the inventory is REBUILT here and not only in ``__init__``.
        The constructor's capability merge runs before any front end can install
        this hook (the TUI resolves its session in a worker and installs it in
        ``_adopt_session``), so the one-time merge always saw ``ask_user=None``
        and ``build_ask_tool`` always returned ``None``: measured on a live TUI
        session, the tools array reaching the provider ended
        ``…, task, wait, jobs, wake, hub`` with no ``ask`` at all, while the
        system prompt instructed the model to use it. The per-turn
        ``_build_tool_context`` is not enough on its own — it decides what a
        tool RUNS against, never whether the tool was advertised.

        Uninstalling drops it again for the same reason it was gated in the
        first place: a tool advertised with no hook behind it can only fail when
        the model calls it.
        """
        self._ask_user = handler
        if handler is not None:
            self._merge_capability_tools(("ask",))
        elif any(tool.name == "ask" for tool in self._tools):
            self.refresh_tools([tool for tool in self._tools if tool.name != "ask"])

    def abort(self, reason: str = "interrupted") -> None:
        """Abort the running turn; the engine emits an aborted agent_end.

        Sticky: between a turn and its post-compaction continuation the live
        signal is None, so a Ctrl+C in that window would otherwise be dropped
        and the agent would run the continuation the user just tried to stop.
        The flag is checked at the top of the continuation drain and pre-aborts
        the next turn's signal.

        A PENDING FORK is withdrawn here too. A fork requested mid-turn waits
        for a boundary that an abort means will never arrive in the shape it was
        asked for, and Esc looking like it stopped everything while a window
        opens minutes later is the surprise this closes — see
        :meth:`cancel_fork`. The host reports it; ``abort`` returns nothing, so
        the caller asks the session what it cancelled.
        """
        self._abort_requested = True
        self.cancel_fork()
        if self._signal is not None:
            self._signal.abort(reason)

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """Cancel every running SUBAGENT and report how many were stopped.

        Deliberately separate from :meth:`abort`, which stops this session's
        own turn. A subagent is a child session with its own turn, its own
        tools and its own spend: aborting the parent's signal does nothing to
        it, so a stopped parent could sit idle while three children carried on
        calling the provider. That is the gap this closes.

        Split from ``abort`` rather than folded into it because the two answer
        different presses. The first Esc stops what the user is watching \u2014 the
        parent's turn \u2014 and a delegated child doing minutes of useful work
        should survive a keypress aimed at the foreground. The second Esc says
        the user meant all of it. Folding them together would make the cheap,
        recoverable stop also the expensive, unrecoverable one.

        ``bash`` jobs are NOT touched, by the same argument in reverse:
        ``background=true`` exists precisely so a build or a deploy outlives
        the turn that started it, and killing one on a keypress meant for the
        agent destroys work the user asked to be insulated from the agent.
        ``jobs cancel`` remains the way to stop those, and session teardown
        still cancels everything.

        Synchronous so a key handler can call it without awaiting: the actual
        cancellation is a task per child, tracked by the session so
        :meth:`dispose` still awaits it. Each child's abort is bridged onto
        its own signal by ``AsyncJobManager.cancel``, so the child settles
        through its own machinery \u2014 persisting the turn it had produced \u2014
        rather than dying mid-await.
        """
        running = self._cancellable_subagents()
        for job in running:
            # Fire-and-forget per child, through the session's own tracker: a
            # child that is slow to unwind must not hold the keystroke, and a
            # cancel that raises must not take its siblings down with it.
            self._spawn_background(self._cancel_job_quietly(job.id, reason))
        return len(running)

    def running_subagents(self) -> int:
        """How many subagents :meth:`cancel_subagents` would stop right now.

        THE one predicate, so a host that offers "esc again to stop N agents"
        and the call that then stops them cannot disagree. They previously did:
        the TUI counted with a filter that excludes jobs parked at the capacity
        gate, while the cancel took every ``running`` row including those — so
        the confirmation contradicted the offer ("1 still running" then
        "stopped 2"), and a press that found ONLY queued children showed
        nothing, armed nothing, and left children the second press would
        happily have cancelled unreachable from the keyboard.

        Queued children are deliberately IN scope: they are delegated work the
        user asked to stop, and one that is merely waiting for a slot is no
        less cancelled by being stopped before it starts.
        """
        return len(self._cancellable_subagents())

    def _cancellable_subagents(self) -> list[Any]:
        """The job rows both the count and the cancel act on. Never raises: a
        keystroke handler and a status line both call into this."""
        try:
            return [
                job
                for job in self.jobs.list()
                if getattr(job, "type", "") == "task" and job.status == "running"
            ]
        except Exception:  # noqa: BLE001 — a count must not take a keypress down
            logger.warning("listing subagent jobs failed", exc_info=True)
            return []

    async def _cancel_job_quietly(self, job_id: str, reason: str) -> None:
        """Cancel one job, logging rather than raising on failure.

        The caller is a keystroke handler with no way to report an error and
        nothing useful to do about one; a child that fails to tear down cleanly
        must not stop its siblings from being cancelled.
        """
        del reason  # the manager stamps its own "cancelled" reason
        try:
            await self.jobs.cancel(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("cancelling subagent job %s failed", job_id, exc_info=True)

    # -- live tool refresh (MCP late-connect / reconnect) ---------------------

    def _set_mcp_startup_sink(self, sink: Callable[[Any], None] | None) -> None:
        """Install the MCP settle/wiring sink through a guarded accessor.

        The attribute itself is private because only two writers should ever
        touch it: the TUI's ``_wire_mcp_status`` (at adoption, possibly BEFORE
        the manager exists — deferred wiring) and ``dispose`` (to drop a sink
        that would otherwise fire into a torn-down app). A setter rather than
        a bare attribute assignment keeps those call sites greppable and lets
        a subclass or reduced host intercept. Noop-safe by contract: the
        wiring's settle path looks the sink up defensively, so a host that
        never installs one simply gets no re-report — the headless default.
        """
        self._on_mcp_startup_settled = sink

    def refresh_tools(self, tools: Sequence[AgentTool]) -> None:
        """Replace the full tool inventory mid-session.

        THE committed hook for MCP ``set_on_tools_changed`` (orchestrator
        MCP-20): the caller passes the merged set (builtins + all currently
        loaded MCP tools) and this swaps it in. The loop reads
        ``context.tools`` fresh on every model call and every tool resolution,
        so the new set is effective from the NEXT model call onward — and even
        mid-turn at the next tool batch — with no restart.
        """
        self._tools = list(tools)
        self._context.tools = self._tools
        if hasattr(self, "_frontend_state_store"):
            self.refresh_frontend_state()

    def set_fallback_tool_resolver(
        self, resolver: Callable[[str], AgentTool | None] | None
    ) -> None:
        """Install a resolver for tool names NOT in the inventory (deferred /
        lazy MCP tools). Wired to ``LoopConfig.resolve_fallback_tool`` so the
        loop can dispatch calls to tools not yet materialized. ``None`` clears
        it."""
        self._fallback_tool_resolver = resolver

    # -- context accounting ---------------------------------------------------

    def _note_usage(self, messages: Sequence[AgentMessage]) -> None:
        """Adopt the newest provider usage in ``messages`` as this
        conversation's latest reading — ONE place for both the compaction
        trigger's figure (``_last_usage``) and the prompt-cache TTL hint.

        Both the post-run scan and the mid-turn boundary hook go through
        here, so the hint can never lag the usage the way it did when only
        the turn boundary moved it (review F9): the hint is what the next
        request of THIS turn (an aside, the advisor) and the NEXT turn's
        first call are stamped with. The hint only ADVANCES on a reported
        count — a wire that omits ``context_tokens`` keeps the previous
        figure rather than blanking it and sending a large context out at
        the 5m TTL by the byte estimate.
        """
        for message in reversed(messages):
            if isinstance(message, Message) and message.usage is not None:
                self._last_usage = message.usage
                if message.usage.context_tokens:
                    self._context_tokens_hint = int(message.usage.context_tokens)
                return

    def restored_usage(self) -> Usage | None:
        """The provider's own last reading for THIS conversation, or ``None``.

        What a resumed front end needs and could not previously get. The status
        band's context segment and its cost segment are both fed by turns that
        end while the app is running, so a resumed session had no source for
        either until the user spent a whole turn — the band opened reporting an
        empty context and no spend for a conversation that might be 40% of the
        way through its window with dollars already on it.

        This is the EXACT figure the provider reported, not an estimate, so a
        host may mark it as such: it is the same number the band would be
        showing had the process never stopped. ``None`` means the transcript
        holds no usage at all (a new session, or a provider that reports none),
        which is why this returns the object rather than a bare int — a caller
        must be able to tell "nothing reported" from "reported zero" and only
        the first justifies falling back to a local estimate.

        Read-only and synchronous: it serves an already-parsed field off the
        replayed history, so a front end may call it on the paint path.
        """
        return self._last_usage

    async def measure_preloaded_context(self) -> int:
        """Tokens the NEXT request carries before the user has typed anything.

        The status line's context reading used to come from one place only:
        ``prompt_tokens`` on a provider's usage response. That is exact, and it
        is unavailable at the one moment a reader most wants it — a session
        opens claiming an empty context when the system prompt, the environment
        block, the skills index and every tool schema are already loaded and
        already spent. On a large tool inventory that is tens of thousands of
        tokens rendered as nothing at all, so the first turn appears to jump
        from 0% to 15% because of a short question.

        What is counted is exactly what :class:`~local_operator.harness.loop`
        puts in a ``ChatRequest``: the system blocks, the serialized tool
        schemas, AND the conversation already in context.

        That last term is not an embellishment, it is the difference between a
        correct reading and a wrong one on every RESUMED session. The name says
        "preloaded", and on a NEW session the history is empty so the two
        readings coincide — which is exactly why the omission survived. A
        ``--resume`` (or ``/resume``, or any reload that keeps the conversation)
        rebuilds ``_context.messages`` from the transcript before this runs, so
        a sum over blocks and schemas alone reports the size of an EMPTY
        conversation for one that may hold hundreds of messages: measured on a
        resumed session whose last provider reading was 402k, the band opened at
        1.7k (0.2%/1M) and stayed there until the next turn happened to end.
        Under-reporting is the dangerous direction — it is the reading a user
        checks to decide whether there is room to keep going, and it claimed a
        whole free window while 40% of it was already spent.

        The history is measured through the SAME renderer a request is built
        from (:meth:`_render_history`), so what is counted is what would
        actually be sent — expired todo reminders dropped, images stripped when
        a provider has refused them — rather than the raw transcript.

        Two costs are deliberately refused, because a status readout must not
        be the most expensive thing a session does before the user speaks:

        - **The tokenizer.** :func:`approx_text_tokens` never loads
          cl100k_base. The exact ruler costs ~43.6 MB RSS and, on a cold cache,
          a NETWORK fetch of the ranks — a cost ``prune_transcript`` and the
          compaction gate both restructure themselves to defer. Paying it in
          every session so the band can read 0.3% would undo that. The ratio
          runs +7% to +17% high depending on how much of the payload is JSON
          punctuation, and the first turn replaces the whole figure with the
          provider's exact count anyway.
        - **The event loop.** Everything that scales with the inventory —
          ``json.dumps`` of each schema as much as the arithmetic over the
          result — happens in the thread. Serializing on the loop and crossing
          only to add up ``len(text) // 4`` was measurably backwards: at 500
          tools the on-loop half cost 3.9 ms against 0.09 ms carried, less than
          the ~0.15 ms the hop itself takes. This runs on the boot path a
          sibling commit cleared of exactly this kind of stall.
        """
        blocks = self._system_blocks()
        if inspect.isawaitable(blocks):
            blocks = await blocks
        resolved = list(blocks)
        # Bind the inventory here, not in the thread. ``refresh_tools`` REBINDS
        # ``self._tools`` rather than mutating it, so this reference stays a
        # coherent snapshot even if an MCP refresh swaps the list mid-count.
        tools = self._tools
        # Rendered on the LOOP for the same reason the schemas are serialized in
        # the thread is right: rendering touches session state (the todo-reminder
        # expiry reads the store, the image degrade reads a flag), and handing
        # mutable session state to a worker thread is how a snapshot tears. The
        # render is a list walk over messages already in memory; the tokenizing
        # of its text is the part that scales, and that still crosses.
        rendered = self._render_history(list(self._context.messages))

        def count() -> int:
            total = sum(approx_text_tokens(text) for text in resolved)
            for tool in tools:
                # Name/description/schema is what a provider serializes per
                # tool. The JSON separators do not match any one vendor's wire
                # format exactly, and no estimate could: this is the same order
                # of magnitude, which is what a percentage needs.
                total += approx_text_tokens(tool.name)
                total += approx_text_tokens(tool.description)
                if tool.parameters:
                    total += approx_text_tokens(json.dumps(tool.parameters, separators=(",", ":")))
            # Counted with THIS method's own ruler, deliberately, rather than
            # with ``estimate_messages_tokens``. That function is the sharper
            # one — it is what compaction plans against — but it reaches for
            # cl100k_base, and the two costs this method's docstring refuses are
            # exactly what that would spend: ~43.6 MB RSS and, on a cold cache, a
            # network fetch of the ranks, on the boot path, before the user has
            # typed anything. Measured on an ordinary session: 58 ms and +41 MB
            # against 0.7 ms and +0 MB for the arithmetic below. The memoization
            # does not rescue it either, because a session that never approaches
            # the compaction threshold never tokenizes at all — so this would not
            # be sharing a cost already paid, it would be creating one.
            #
            # The terms mirror ``_compute_tokens`` so the shape of what is
            # counted matches the sharper ruler even though the ruler differs:
            # text, a flat charge per image, and each tool call's name plus its
            # serialized arguments. Dropping the last two would understate a
            # resumed vision or tool-heavy session by its largest terms.
            for message in rendered:
                for block in message.content:
                    if isinstance(block, TextContent):
                        total += approx_text_tokens(block.text)
                    else:
                        total += IMAGE_TOKEN_ESTIMATE
                for call in message.tool_calls:
                    total += approx_text_tokens(call.name)
                    total += approx_text_tokens(
                        call.raw_arguments or json.dumps(call.arguments, sort_keys=True)
                    )
            return total

        return await asyncio.to_thread(count)

    # -- events ---------------------------------------------------------------

    def _flush_frontend_jobs(self) -> None:
        """Coalesce a burst of child trajectory/progress mutations per loop tick."""
        self._frontend_jobs_refresh_scheduled = False
        store = getattr(self, "_frontend_state_store", None)
        if store is not None:
            store.refresh_jobs(self)

    @property
    def frontend_state(self):  # type: ignore[no-untyped-def]
        """Current canonical state for any full terminal frontend."""
        return self._frontend_state_store.state

    def subscribe_frontend(self, handler):  # type: ignore[no-untyped-def]
        """Atomically refresh, snapshot and subscribe on the session loop.

        The refresh must go through the PUBLISHING path: silently replacing
        state would hand this joiner a different state at the same sequence
        number an earlier subscriber already holds, breaking the exact-`+1`
        gap check every client uses to detect transport loss. ``initial=True``
        is reserved for store construction, before any subscriber exists.
        """
        self._frontend_state_store.refresh_from_session(self)
        return self._frontend_state_store.subscribe(handler)

    def refresh_frontend_state(self) -> None:
        """Publish non-event source changes through the canonical contract.

        Guarded because the attachment restore (#301) re-attaches the stored
        team/agent DURING construction, before the store exists; those
        mutations are captured by the store's own construction snapshot, so
        skipping the publish here loses nothing.
        """
        store = getattr(self, "_frontend_state_store", None)
        if store is not None:
            store.refresh_from_session(self)

    def refresh_frontend_usage(self) -> None:
        """Refresh restored usage without scanning transcript/jobs/tool schemas."""
        self._frontend_state_store.refresh_restored_usage(self)

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register an event handler; returns an unsubscribe callable. Sync or
        async handlers are called in registration order; one raising never
        breaks the others."""
        self._handlers.append(handler)

        def _unsubscribe() -> None:
            try:
                self._handlers.remove(handler)
            except ValueError:
                pass

        return _unsubscribe

    async def _stream_notice(
        self,
        text: str,
        kind: Literal["info", "warning", "error"] = "warning",
    ) -> None:
        """Bridge provider-routing diagnostics onto the session event stream."""
        await self._emit(NoticeEvent(text=text, kind=kind))

    async def _on_route_settled(self, target: Any, reason: str) -> None:
        """The stream fn's effective route moved; record it and tell the host.

        ``target`` is the pinned ``FallbackTarget`` (selector + optional
        effort) or ``None`` when requests returned to the selected model. Three
        jobs, in order: derive the fallback's own spec (metadata belongs to the
        TARGET model — a display that keeps the primary's context window under
        a fallback's name misreports both), persist the edge so a resume comes
        back on the right model, and emit the event the front end repaints its
        model display from.
        """
        if target is None:
            self._active_fallback = None
            self._active_route = None
            await self._persist_active_route(self._model)
            await self._emit(
                ModelChangeEvent(
                    provider=self._model.provider,
                    model_id=self._model.model_id,
                    effort=self._model.reasoning_effort,
                    reason=reason,
                    is_fallback=False,
                    context_window=int(self._model.context_window),
                )
            )
            # Tell the model it is back on its primary. Not transient: the
            # session has returned to its selected model for the foreseeable
            # future, unlike the outbound fallback below.
            await self.journal_model_switch(
                f"{self._model.provider}/{self._model.model_id}",
                reason=reason or "returned to primary model",
                transient=False,
            )
            return
        selector = str(target.selector)
        target_effort = getattr(target, "effort", None)
        spec = self._spec_for_route(selector, target_effort)
        if spec is None:
            # An unresolvable selector must not take the session down mid-turn;
            # the fallback still serves, the display just cannot follow it.
            logger.warning("could not resolve fallback spec for %r", target.selector)
            return
        self._active_fallback = spec
        self._active_route = (selector, target_effort)
        await self._persist_active_route(self._model)
        await self._emit(
            ModelChangeEvent(
                provider=spec.provider,
                model_id=spec.model_id,
                effort=spec.reasoning_effort,
                reason=reason,
                is_fallback=True,
                context_window=int(spec.context_window),
            )
        )
        # Tell the model it is now answering on a FALLBACK. Transient: the
        # session may return to its primary at a later boundary (the target=None
        # branch above), and the model should know the change may not persist so
        # it does not, for example, record the fallback as its identity.
        await self.journal_model_switch(
            f"{spec.provider}/{spec.model_id}",
            f"{self._model.provider}/{self._model.model_id}",
            reason=reason,
            transient=True,
        )

    def _spec_for_route(self, selector: str, effort: str | None) -> ModelSpec | None:
        """The fallback target's own ``ModelSpec``, or None when unresolvable.

        Through :func:`~local_operator.providers.failover.spec_for_target`
        because that is the SAME derivation the failover driver uses to build
        the request — deriving the display spec any other way is how the band
        and the wire end up disagreeing about effort or context window.
        """
        try:
            from local_operator.providers.failover import (
                FallbackTarget,
                spec_for_target,
            )

            return spec_for_target(self._model, FallbackTarget(selector, effort))
        except Exception:  # noqa: BLE001 — display state is never worth a broken turn
            return None

    async def _emit(self, event: AgentEvent) -> None:
        # Fold before fan-out: a client joining from an event handler observes a
        # snapshot that already contains this event, never an off-by-one view.
        store = getattr(self, "_frontend_state_store", None)
        if store is not None and (self._has_ui or store.has_subscribers):
            store.observe_event(self, event)
        for handler in list(self._handlers):
            try:
                outcome = handler(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                logger.warning("event handler failed for %s", event.type, exc_info=True)

    def _emit_nowait(self, event: AgentEvent) -> None:
        """Fire-and-forget ``_emit`` for a caller that cannot await.

        The resume catch-up send (``_handle_missed_wakes``) is synchronous —
        it runs at the head of the boot pump — but a front end still needs the
        delivery receipt to paint its wake line. Handlers here are the event
        controller's ``_post`` (a sync queue push), so routing the coroutine
        through the background-task machinery delivers it on the next loop
        pass, ahead of the turn the same method spawns.
        """
        self._spawn_background(self._emit(event))

    # -- turn machinery --------------------------------------------------------

    async def _prompt_messages(self, initial: list[AgentMessage]) -> None:
        """Shared turn runner for wake deliveries (prompt() owns its own lock
        handling so it can REJECT reentrants instead of queueing)."""
        if self._disposed:
            raise RuntimeError("session is disposed")
        async with self._turn_lock:
            # Same shell-receipt boundary as prompt(): a wake turn must not
            # build from context that omits a command already visible in TUI.
            await self._flush_shell_records()
            # Same boundary, same reason: a notice parked by a turn that ended
            # without reaching a steering drain (an abort mid-batch) rejoins
            # here, before this turn builds its request.
            self._flush_context_journal()
            # Same flush as prompt(): a wake-delivery turn is still a prompt
            # path, and the catch-up must not be stranded behind it.
            self._handle_missed_wakes()
            await self._run_turn_pipeline(initial)

    async def _run_turn_pipeline(
        self,
        initial: list[AgentMessage],
        *,
        admitted: asyncio.Future[None] | None = None,
        admitted_id: str | None = None,
        producer_command_id: str | None = None,
    ) -> None:
        """One turn + its auto-continuations. Caller holds ``_turn_lock``.

        A post-compaction continuation is a CONTINUATION of the same logical
        run, not a new one: compaction happens after the loop has already
        yielded its ``agent_end``, so forwarding that end (and the next run's
        ``agent_start``) would tell every UI the task finished and then started
        again. The pipeline therefore holds the boundary events and emits
        exactly one ``agent_start`` / ``agent_end`` pair per user prompt, with
        ``compaction_start`` / ``compaction_end`` and further turns in between.
        The generation stamp on the emitted end is the one from the start that
        opened the run, so the TUI's supersede guard still pairs them.
        """
        # Re-arm the todo guardrail: a fresh user message may well be the answer
        # a stalled list was waiting for, so the latch must not carry over. It is
        # reset HERE and not in `_run_turn` on purpose — `_run_turn` also runs
        # post-compaction continuations of this SAME user turn, and re-arming
        # there would re-nudge an unchanged list the model already declined.
        self._todo_reminder_fingerprint = None
        begin_message = getattr(self._stream_fn, "begin_message", None)
        if callable(begin_message):
            begin_message()
        self._turn_task = asyncio.current_task()
        try:
            await self._run_turn(
                initial,
                admitted=admitted,
                admitted_id=admitted_id,
                producer_command_id=producer_command_id,
            )
            await self._drain_continuation()
        finally:
            self._turn_task = None
            await self._flush_held_end()
            # A bang-mode command may have completed while this turn owned the
            # provider message list. Persist it before releasing `_turn_lock`,
            # when no next prompt can build from a half-updated conversation.
            await self._flush_shell_records()
            # Likewise for a journal notice parked mid-turn. The tool batch is
            # closed by now, so the append is legal, and doing it here means an
            # aborted turn still hands the notice to the next one instead of
            # stranding it until the session is disposed.
            self._flush_context_journal()

    async def _flush_held_end(self) -> None:
        """Emit the boundary event the pipeline was holding, if any."""
        held = self._held_end
        self._held_end = None
        context_tokens = self._held_context_tokens
        self._held_context_tokens = None
        if held is None:
            return
        generation = self._logical_generation or held.generation
        self._logical_generation = None
        changes: dict[str, Any] = {}
        if held.generation != generation:
            changes["generation"] = generation
        if context_tokens is not None:
            changes["context_tokens"] = context_tokens
        await self._emit(held if not changes else held.model_copy(update=changes))

    async def _run_turn(
        self,
        initial: list[AgentMessage],
        *,
        admitted: asyncio.Future[None] | None = None,
        admitted_id: str | None = None,
        producer_command_id: str | None = None,
    ) -> None:
        """One loop run + persistence. Caller holds ``_turn_lock``."""
        if self._wake.needs_rearm:
            # HC-20: the scheduler could not arm without a running loop at
            # construction; the first turn (with a loop) re-arms via pump().
            self._handle_missed_wakes()
            await self._wake.pump()
        self._is_streaming = True
        self._generation += 1  # monotonic; stamped on start AND end events
        self._last_activity_ms = int(time.time() * 1000)
        signal = AbortSignal()
        self._signal = signal
        if self._abort_requested:
            # A Ctrl+C landed in the gap between turns (signal was None);
            # honour it on the fresh signal instead of running the
            # continuation the user tried to stop.
            signal.abort("interrupted")
        try:
            for message in initial:
                await self._transcript.append_message(
                    message,
                    producer_command_id=(
                        producer_command_id if message.id == admitted_id else None
                    ),
                )
                if admitted is not None and message.id == admitted_id and not admitted.done():
                    # The append completed under Transcript's fsync boundary;
                    # only now may a producer discard its retained command.
                    admitted.set_result(None)
                # Announce USER turns to every subscriber. The loop only emits
                # MessageStartEvent for ASSISTANT messages, so without this a
                # user prompt — from any front end — never reaches the other
                # surfaces: a TUI prompt was invisible on the phone, a phone
                # prompt invisible in the TUI. Emitting here, at the append
                # point, is the single source both read. Wake/continuation
                # internals are CustomMessage, so this stays user-authored only.
                # The auto-continuation prompt is the one user-shaped Message
                # that is harness chrome, not the user's words: announcing it
                # would stack "context was just compacted" user rows on every
                # front end for prompts the human never typed (the same
                # exemption LOOP_PROMPT gets in the TUI). Matching is by text
                # equality, so a user who typed the continuation sentence
                # verbatim would lose their announcement — vanishingly
                # unlikely, and the TUI registry documents its equivalent
                # inherent limit.
                if (
                    isinstance(message, Message)
                    and message.role == "user"
                    and message.text != _CONTINUATION_PROMPT
                ):
                    await self._emit(MessageStartEvent(message=message))

            blocks = self._system_blocks()
            if inspect.isawaitable(blocks):
                blocks = await blocks
            self._context.system_blocks = list(blocks)
            self._context.tool_context = self._build_tool_context()

            config = LoopConfig(
                model=self._model,
                # Re-read per provider call, so a ``set_model`` landing while
                # this turn is running reaches its NEXT call instead of
                # waiting for the turn to end (see ``set_model``).
                get_model=lambda: self._model,
                # The volatile tail contains the live /goal, /team and /agent
                # instructions. Re-read it per provider step so a setting changed
                # while a tool runs reaches the next call in THIS turn rather than
                # waiting for another user message. The loop keeps the turn-start
                # snapshot as a fallback if this host resolver ever fails.
                get_system_blocks=self._system_blocks,
                # Cross-turn seed for the prompt-cache TTL hint: the loop stamps
                # it on the run's first request and then prefers the counts its
                # own calls report. Lives here — not on the shared stream fn —
                # so a subagent's calls cannot contaminate this session's hint;
                # see ``LoopConfig.get_context_tokens_hint``.
                get_context_tokens_hint=lambda: self._context_tokens_hint,
                convert_to_llm=self._render_history,
                stream_fn=self._stream_fn,
                get_steering_messages=self._drain_steering,
                has_steering_messages=lambda: not self._steering_queue.empty(),
                has_urgent_steering_messages=self._has_urgent_steering,
                # Rides the SAME interrupt poll steering uses, so a fork
                # requested mid-tool reaches its boundary in ~250 ms instead of
                # after a long `wait`/`bash`/MCP call. It cannot skip the
                # batch's remaining calls; see LoopConfig.has_pending_fork.
                has_pending_fork=self.has_pending_fork,
                get_aside_messages=self._drain_asides,
                get_follow_up_messages=self._todo_continuation,
                resolve_fallback_tool=self._fallback_tool_resolver,
                # Redact stored credential values out of every tool result
                # before the message lands in the transcript. The store is
                # in-memory and session-scoped, so this is the one place a
                # credential can be turned back into plain text for the model.
                redact_tool_result=(
                    self._variables.redact if self._variables is not None else None
                ),
                interrupt_mode="immediate",
                on_turn_end=self._on_turn_end,
            )

            new_messages: list[AgentMessage] = []
            async for event in self._loop.run(
                initial, self._context, config, signal, generation=self._generation
            ):
                if isinstance(event, AgentStartEvent):
                    if self._logical_generation is None:
                        # First run of the pipeline: this start opens the run
                        # the UI sees, and its generation stamps the eventual
                        # end. Continuation runs re-enter here and are silent.
                        self._logical_generation = event.generation
                        await self._emit(event)
                    else:
                        # A continuation's provider receipt is newer than the
                        # estimate produced between runs. Demote that estimate
                        # as soon as the later request begins so the eventual
                        # held end can expose its own authoritative occupancy.
                        self._held_context_tokens = None
                    continue
                if isinstance(event, AgentEndEvent):
                    new_messages = list(event.messages)
                    if event.aborted or event.error:
                        # A failed or interrupted run is a real boundary: never
                        # hold it behind a compaction that may not happen. The
                        # end is re-stamped with the generation of the start
                        # the UI saw (the continuation's own stamp would split
                        # the pair), and the continuation queue is dropped so
                        # no further run can open a second boundary inside the
                        # same prompt (§B).
                        if event.error is not None:
                            await self._degrade_if_image_rejected(event.error)
                            # Journal WHY the run died after persistence: the
                            # model (this session or a resumed one) must see
                            # the failure, not just the UI.
                            self._pending_incident = event.error
                        self._abort_requested = True
                        self._continuation_queue.clear()
                        self._held_end = None
                        generation = self._logical_generation or event.generation
                        self._logical_generation = None
                        await self._emit(
                            event
                            if event.generation == generation
                            else event.model_copy(update={"generation": generation})
                        )
                    else:
                        # Held until _maybe_compact has had its say; the
                        # pipeline flushes it if no continuation is queued.
                        self._held_end = event
                    continue
                is_todo_end = isinstance(event, ToolExecutionEndEvent) and event.tool_name == "todo"
                if is_todo_end:
                    # The tool has already mutated its store when this event is
                    # yielded. Persist before async subscriber fan-out so a
                    # cancellation from a handler cannot expose live todo state
                    # that resume cannot recover. The fingerprint guard makes
                    # failed/view/no-op todo calls a zero-write comparison.
                    await self._maybe_persist_todos()
                await self._emit(event)
                if is_todo_end and self._job_id is not None and self._subagent_comms is not None:
                    self._subagent_comms.notify_detail_persisted(self._job_id)

            # Track the latest provider usage for compaction trigger math (and
            # the TTL hint that rides with it).
            self._note_usage(new_messages)
            self._last_activity_ms = int(time.time() * 1000)
            # The run just completed provider round-trips; the idle flush
            # measures provider-cache age from this stamp, not turn
            # bookkeeping (which would always read ~0 and kill the flush).
            self._last_provider_request_ms = int(time.time() * 1000)

            # Persist everything the turn produced (initial messages were
            # written before the run). Deduplicated by id rather than by
            # identity against ``initial``, because the mid-turn compaction
            # gate may already have flushed part of this run to get a
            # replayable cut target; re-appending those would resurrect
            # messages after the compaction entry that superseded them.
            await self._persist_new_messages(new_messages)

            # The SECOND fork drain point, and it is not redundant with the one
            # in ``_on_turn_end``. That hook fires only when the loop will
            # CONTINUE, so a fork requested during the final model turn of a run
            # — the common case of "type /fork while it is finishing up" —
            # never reaches it. Without this line that fork silently waits for
            # the user's next turn, which reads as the command doing nothing.
            # Placed after the persist above for the same reason as the other
            # site: the clone copies what is on disk.
            await self._drain_pending_fork()

            # Snapshot the todo list when it moved this turn. Guarded by the
            # same full-list fingerprint the continuation guardrail uses, so an
            # unchanged list costs one tuple comparison and no transcript write,
            # while any add/done/block/drop/init lands on disk for the next
            # resume. Placed on the normal path (a turn that raised past here
            # still has last turn's snapshot; the next clean turn re-writes it).
            await self._maybe_persist_todos()
            # Spend must survive owner death at the same durability boundary as
            # the messages it describes. The checkpoint is replacement state,
            # never an additive delta, so takeover cannot double it. A headless
            # store that observed nothing (no UI, no attach subscriber) skips
            # the write: it holds only the restored durable state, and
            # re-appending that would at best duplicate and at worst — before
            # ``from_checkpoint`` — clobber the richer checkpoint a TUI wrote.
            if self._has_ui or self._frontend_state_store.has_subscribers:
                await self._frontend_state_store.checkpoint(self._transcript)

            # Child events reach the shared comms watcher before either durable
            # append. Notify only after messages AND todos are stable, including
            # provider-error turns that emit no later terminal event.
            if self._job_id is not None and self._subagent_comms is not None:
                self._subagent_comms.notify_detail_persisted(self._job_id)

            pending_incident = self._pending_incident
            self._pending_incident = None
            if pending_incident:
                await self.journal_incident(pending_incident)

            await self._maybe_compact()
        finally:
            # LAST-RESORT durability. The persistence above runs only on the
            # normal path: an exception out of the loop (the
            # "model turn produced no assistant message" RuntimeError, a
            # provider client raising past the stream handler), a
            # ``CancelledError`` from Ctrl+C or dispose, or a steering-driven
            # teardown all skip it entirely, and everything the run completed
            # died in memory with it. The per-boundary flush covers whole tool
            # batches; this covers the tail produced after the last boundary,
            # including the assistant message whose own failure ended the run.
            #
            # Best-effort and never raising: this is a ``finally`` on the way
            # out of a turn that may already be unwinding, and a transcript
            # write that fails must not replace the original exception (which
            # is what the caller and the incident journal need to see).
            await self._persist_progress(self._context.messages)
            self._signal = None
            self._is_streaming = False

    async def _drain_continuation(self) -> None:
        """Run queued auto-continuation prompts (post-compaction resume) as
        sequential turns inside the same lock hold, capped at
        ``_MAX_CONTINUATIONS`` — the remainder is dropped with a warning
        notice so a thrashing recovery band cannot re-prompt forever."""
        continuations = 0
        while self._continuation_queue and not self._disposed:
            if self._abort_requested:
                # Ctrl+C landed in the gap between turns; the continuation
                # the user interrupted must not run.
                self._continuation_queue.clear()
                return
            if continuations >= _MAX_CONTINUATIONS:
                dropped = len(self._continuation_queue)
                self._continuation_queue.clear()
                logger.warning(
                    "auto-continuation cap (%d) reached; dropping %d queued continuation(s)",
                    _MAX_CONTINUATIONS,
                    dropped,
                )
                await self._emit(
                    NoticeEvent(
                        text=(
                            f"Auto-continuation limit ({_MAX_CONTINUATIONS}) reached; " "stopping."
                        ),
                        kind="warning",
                    )
                )
                return
            message = self._continuation_queue.pop(0)
            await self._run_turn([message])
            continuations += 1

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any] | None:
        """Route a fire-and-forget coroutine through the session task group
        when one is open (wake deliveries, aside persistence); otherwise fall
        back to ``ensure_future``. Every spawned task is tracked so
        :meth:`dispose` can cancel and await it. After dispose nothing is
        spawned, so a late wake delivery cannot raise into an unobserved task.

        Returns the Task so a caller that later needs to CANCEL this one
        specifically (the ceiling compaction pass, issue #413) can, without
        walking ``_background_tasks``. ``None`` after dispose: nothing was
        spawned.

        The coroutine is wrapped so its failure is logged, never raised: an
        exception escaping into a TaskGroup would cancel every sibling task.
        """
        if self._disposed:
            # Close the coroutine we were handed rather than dropping it: a
            # caller building ``self._spawn_background(self._coro())`` has
            # already created it, and a disposed session that merely returned
            # left it unawaited — an "coroutine was never awaited" warning, and
            # under a background subagent runner racing dispose it is a real
            # path (the runner schedules a roster persist just as teardown
            # flips this flag).
            coro.close()
            return None

        started = False

        async def _guarded() -> None:
            nonlocal started
            started = True
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("background session task failed", exc_info=True)

        if self._task_group is not None:
            task = self._task_group.create_task(_guarded())
        else:
            wrapper = _guarded()
            try:
                task = asyncio.ensure_future(wrapper)
            except RuntimeError:
                # No running loop: ``ensure_future`` raises, and BOTH coroutines
                # are already built — the wrapper here and the ``coro`` the
                # caller handed us. Closing them is the same courtesy the
                # disposed branch above pays, and for the same reason: an
                # un-awaited coroutine is reported by asyncio at GC time,
                # blaming the session for work it never agreed to run. Re-raised
                # afterwards so a caller that genuinely needs a loop still hears
                # about it; the callers for which this is merely decoration
                # (``_push_browser_title``) catch it themselves.
                wrapper.close()
                coro.close()
                raise
        self._background_tasks.add(task)

        def _on_done(finished: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(finished)
            # If dispose cancelled this task before ``_guarded`` ever ran (it
            # was still pending), the wrapper's ``try`` never entered and the
            # INNER ``coro`` was created but never awaited — Python reports it
            # as "coroutine was never awaited". Closing it here, from the one
            # callback that fires for every task including a cancelled-pending
            # one, is what suppresses that. ``started`` is the discriminator:
            # once the wrapper began, ``await coro`` drove the coroutine and a
            # late ``close()`` would be redundant (and is skipped).
            if not started:
                coro.close()

        task.add_done_callback(_on_done)
        return task

    def _build_tool_context(self) -> ToolContext:
        # This context is REBUILT on every turn, so anything that must outlive
        # a turn is owned by the session and injected here. ``wake_scheduler``
        # is a declared ToolContext field: the wake tool's createIf check and
        # executor read it off the context, and a session without a scheduler
        # must not advertise the tool at all. ``browser`` is the same shape —
        # the surface handle has to survive to the next turn for browsing to
        # work, and for teardown to be able to close the tab.
        return ToolContext(
            cwd=self._cwd,
            session_id=self._session_id,
            # Re-read the live holder every turn so generated, user-set, and
            # resumed titles reach display-only browser metadata after renames.
            # On a CHILD this resolves to the parent's title (see
            # :meth:`_display_session_name`), which is the only conversation
            # name a subagent has.
            session_name=self._display_session_name(),
            # Per-turn is not enough on its own: this context is a SNAPSHOT
            # taken once at turn start, and the naming errand lands a second or
            # two into the FIRST turn — so a browse in that turn read "" even
            # after the title existed, and the tab group latched the fallback
            # label for the life of the tab. The callable re-reads the holder at
            # tool-call time. Display-only, like ``session_name`` itself.
            session_name_provider=self._display_session_name,
            agent_id=self._agent_id,
            # The delegated name, on a subagent only. Empty on every top-level
            # session, which is what keeps ``_browser_subagent_label``'s
            # ``job_id`` discriminator honest.
            job_label=self._job_label,
            has_ui=self._has_ui,
            resolve_internal_url=self._skill_resolver,
            request_approval=None if self._yolo else self._request_approval,
            ask_user=self._ask_user,
            wake_scheduler=self._wake,
            on_todos_changed=self.refresh_frontend_state,
            browser=self._browser,
            subagent_launcher=self._launch_subagent,
            jobs=self.jobs,
            peer_arrival=self._peer_arrival,
            subagent_comms=self.subagent_comms,
            variables=self._variables,
            # The ``ask`` tool stores secret answers straight into the store;
            # this hook is what lets the session ANNOUNCE the new key to the
            # model instead of leaving it to notice the prompt tail (the
            # failure behind session 835fbcafdc27). Re-bound every turn for
            # the same reason the rest of this context is rebuilt.
            journal_credential=self.journal_credential_change,
            job_id=self._job_id,
            agent_registry=self.agent_registry,
            team_registry=self.team_registry,
            delegated_tools={
                tool.name: tool for tool in self._tools if tool.name.startswith("mcp__")
            },
        )

    def _launch_subagent(
        self, label: str, prompt: str, *, agent: str = "task", effort: str | None = None
    ) -> str:
        """Register one one-shot child run on this session's job manager.

        The production caller of :func:`run_subagent`: spawn the child against
        ``self.jobs`` (so its lifecycle lands in this session's job manager and
        the parent's dispose cancels it), reusing this session's own bit of
        the child wiring. Installed on the ``ToolContext`` as
        ``subagent_launcher`` every turn so the ``task`` tool can start a
        child. Returns the job id.

        ``agent``/``effort`` select the tier (see ``SubagentLauncher``):
        effort resolves through ``values.subagents.models`` (``lo``/``med``/
        ``hi`` -> ``provider/model-id``), so a scout or a cheap bulk task runs
        on the model the operator picked for that job, not the session's
        default. An unresolvable tier falls back to the parent's model with a
        warning — a delegation must not fail because a config key is stale.
        """
        model_spec = self._resolve_subagent_model(agent, effort)
        return run_subagent(
            label=label,
            prompt=prompt,
            parent_session=self,
            jobs_manager=self.jobs,
            model_spec=model_spec,
            agent=agent,
            # Recorded on the job for the title/band to name the tier. The spec
            # above already carries the resolved MODEL; this carries the LEVEL,
            # which the spec cannot reconstruct (see ``run_subagent``).
            effort=effort,
        )

    def _resolve_subagent_model(self, agent: str, effort: str | None) -> ModelSpec | None:
        """Effort tier -> ModelSpec via config; None keeps the parent's model.

        Precedence: an explicit ``effort`` on the launch beats the role's own
        default, which beats the session model. That order is what lets a role
        say "this job is usually cheap" while a caller who knows this instance
        is hard can still pay for the better model. No role hardcodes a tier
        here: the packaged seeds deliberately pin none, so a delegated child
        inherits the session's model unless the OPERATOR chose a tier — a
        shipped default that silently downgraded review quality could not be
        traced to anything the operator decided.
        """
        wanted = effort
        if wanted is None and agent and agent != "task":
            try:
                from local_operator.agent_profiles import resolve_profile

                profile = resolve_profile(agent, registry=self.agent_registry)
            except Exception:  # noqa: BLE001 - tier lookup must not fail a spawn
                profile = None
            if profile is not None:
                wanted = profile.effort
        if wanted is None:
            return None
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            raw = ConfigManager(config_dir()).get_config_value("subagents", None)
            models = raw.get("models", {}) if isinstance(raw, dict) else {}
            selector = models.get(wanted)
            if not selector:
                return None
            provider, _, model_id = str(selector).partition("/")
            if not model_id:
                logger.warning("subagents.models.%s=%r lacks provider/model", wanted, selector)
                return None
            from local_operator.model.configure import build_model_spec

            return build_model_spec(provider, model_id)
        except Exception:  # noqa: BLE001 — stale config must not fail a spawn
            logger.warning(
                "subagent model tier %r could not be resolved; using session model", wanted
            )
            return None

    def _append_or_park_journal(self, message: CustomMessage) -> None:
        """Put a journal notice on the live context, or park it for a boundary.

        The live context may not take a non-tool message while a turn owns it.
        ``AgentLoop`` appends the assistant message the moment the model turn
        ends and appends the tool results only once the WHOLE batch returns
        (``_append_results``), so for the entire duration of a tool batch the
        list ends in an assistant whose ``tool_calls`` have no answers.
        Splicing a notice in there produces
        ``assistant(tool_use) -> user -> tool_result``, which every provider
        rejects ("`tool_use` ids were found without `tool_result` blocks
        immediately after") — and because nothing repairs the live list, the
        session re-sends that prefix on every later turn and is bricked until
        it is restarted. Observed live: a ``/model`` press mid-batch killed a
        session outright, including a bare "Continue".

        BOTH conditions are tested, for the reason :meth:`prompt` spells out:
        ``_is_streaming`` covers only ``_run_turn``, while ``_turn_lock`` spans
        the whole pipeline including a post-compaction auto-continuation.
        ``record_shell`` and ``adopt_aside`` guard on the same pair.

        The delay is accepted, not a compromise to be "fixed" later by
        re-splicing: the notice is advisory, the env block's ``Model:`` line
        already carries the live identity, and the transcript write has already
        happened, so the worst case is that a switch NOTICE lands at the end of
        a long tool batch instead of inside it. Re-splicing to make it prompt
        is precisely the bug above.
        """
        if self._is_streaming or self._turn_lock.locked():
            self._pending_context_journal.append(message)
            return
        self._context.messages.append(message)

    def _drain_context_journal(self) -> list[CustomMessage]:
        """Pop every parked journal notice, oldest first.

        Synchronous and total: the transcript write already happened at park
        time, so there is nothing here that can fail and nothing to retry —
        unlike ``_flush_shell_records``, which pops only after a successful
        write. Split out from its one caller, ``_flush_context_journal``, so
        the drain is testable on its own.
        """
        parked = self._pending_context_journal
        self._pending_context_journal = []
        return parked

    def _flush_context_journal(self) -> None:
        """Fold parked journal notices into the live context at a safe boundary.

        A parked notice must never be silently dropped: it is the only thing
        telling the model it was switched or that the last run failed.
        """
        parked = self._drain_context_journal()
        if parked:
            self._context.messages.extend(parked)

    async def journal_incident(self, raw: str) -> None:
        """Persist and surface WHY the session last failed.

        The failover cascade rotates credentials and models and its notices
        reach the UI, but without this the MODEL never learned the
        difference between "quota exhausted" and "my own bug" — the next
        prompt (and a resumed session) resumed blind. The incident is
        classified (rate-limit / auth / provider / network / context / MCP),
        appended to the LIVE context so the very next turn sees it, and
        persisted so ``--resume`` replays it.
        """
        from local_operator.incidents import format_incident_message

        if self._disposed or not raw:
            return
        text = format_incident_message(raw, self._model.provider, self._model.model_id)
        message = CustomMessage(
            custom_type=SESSION_INCIDENT_MESSAGE_TYPE,
            attribution="system",
            details={"text": text, "raw": raw[:1000]},
        )
        try:
            await self._transcript.append_message(message)
            self._append_or_park_journal(message)
        except OSError:
            logger.warning("could not journal session incident", exc_info=True)

    async def journal_model_switch(
        self,
        new_label: str,
        previous_label: str = "",
        *,
        reason: str = "",
        transient: bool = False,
    ) -> None:
        """Make a model change visible to the MODEL, not just the UI.

        A deliberate ``/model`` switch and a failover fallback both already
        emit a ``ModelChangeEvent`` the front end repaints from, and the
        selection is journalled as ``SELECTED_MODEL_CUSTOM_TYPE`` so a resume
        restores it — but none of that reaches the model's own context. Without
        this, a model that has just been switched onto (or a subagent whose run
        failed over to a different model) keeps reasoning as though it were the
        previous model: it can misreport which model it is, assume the wrong
        context window, or name the wrong model in a byline. This appends a
        ``session_model_switch`` message to the live context so the next turn
        sees the change. A non-transient (deliberate, or return-to-primary)
        record is also persisted so a resumed session replays it; a TRANSIENT
        fallback record is live-only, because a fallback that outlives the
        process is stale on resume (see ``_is_persistable_message``, review R1).

        Deduplicated against the last switch record so the two edges that can
        both fire for one change (``set_model`` and a route-settled event) do
        not double-announce: a repeat naming the same ``new_label`` with the
        same ``transient`` flag is dropped.

        Called on the loop, but not synchronously ordered against the next turn:
        ``set_model`` spawns this in the background. The env-block ``Model:``
        line already carries the live identity, so the worst case is that the
        "you just switched" NOTICE lands one turn late, never that the model's
        identity is wrong (review R2).
        """
        from local_operator.incidents import format_model_switch_message

        if self._disposed or not new_label:
            return
        if (new_label, transient) == self._last_model_switch_announced:
            return
        self._last_model_switch_announced = (new_label, transient)
        text = format_model_switch_message(
            new_label, previous_label, reason=reason, transient=transient
        )
        message = CustomMessage(
            custom_type=SESSION_MODEL_SWITCH_MESSAGE_TYPE,
            attribution="system",
            details={
                "text": text,
                "new_label": new_label,
                "previous_label": previous_label,
                "transient": transient,
            },
        )
        try:
            # Persist only when the record should survive a resume; a transient
            # fallback is live-context-only (see _is_persistable_message).
            if _is_persistable_message(message):
                await self._transcript.append_message(message)
            self._append_or_park_journal(message)
        except OSError:
            logger.warning("could not journal model switch", exc_info=True)

    def journal_credential_change(
        self,
        key: str,
        *,
        action: str = "stored",
        replaced: bool = False,
    ) -> None:
        """Make a mid-session credential change visible to the MODEL.

        Storing or forgetting a credential already updates the store (bash
        injection follows) and the ``<session-credentials>`` block in the
        volatile prompt tail — but nothing in the live conversation said so.
        The model has no reason to re-read a tail it already read, so the
        operator's ``I just added the API key`` left it guessing names until
        it noticed the tail by accident (session 835fbcafdc27: ten minutes
        between the store and the model finding the real key name). This
        appends a ``session_credential`` message to the LIVE context that
        renders as a user turn, naming the KEY only — the value never rides a
        message the provider sees. The record is deliberately NOT persisted:
        credentials are process-memory-only, so a replayed announcement would
        assert an env var a restarted session does not have (review round 1,
        R2).

        SYNC, unlike :meth:`journal_model_switch`: the TUI's
        ``/credential`` flow and the ``ask`` tool both call this from the
        loop, and the store write has ALREADY happened by the time they do —
        a fire-and-forget task could be reordered after a turn that starts
        before the next loop tick, which is exactly the window where the
        model asks "what key?" one turn later. The transcript write is the
        only await-shaped step, so this parks it on the same background
        machinery as the rest of the journal (``_spawn_background``) while
        the live-context append is immediate.

        Called AFTER a successful store/forget, never before: the
        announcement must not claim a credential the store refused (empty
        value, empty key).
        """
        from local_operator.incidents import format_credential_message

        if self._disposed or not key:
            return
        text = format_credential_message(key, action=action, replaced=replaced)
        message = CustomMessage(
            custom_type=SESSION_CREDENTIAL_MESSAGE_TYPE,
            attribution="system",
            details={
                "text": text,
                "key": key,
                "action": action,
                "replaced": replaced,
            },
        )
        # Live context ONLY (review round 1, R2). The announcement asserts a
        # live capability — "$KEY is injected into every bash command" —
        # against a store that is process-memory-only and rebuilt empty on
        # every start. Persisting it meant a resumed session replayed the
        # claim as a user turn for an env var bash no longer has. The
        # mid-session discovery problem this fixes (the model never re-reads
        # the prompt tail) is fully served by the live append; resume-time
        # discovery is served by the ``<session-credentials>`` tail block,
        # which is rebuilt from the live store each turn and so correctly
        # shows nothing after a restart. Mirrors the transient model-switch
        # split (``_is_persistable_message``).
        self._append_or_park_journal(message)

    def _on_mcp_incident(self, server: str, reason: str) -> None:
        """MCP manager hook (breaker trips): journal without blocking the
        manager's reconnect loop."""
        self._spawn_background(self.journal_incident(f"MCP server '{server}': {reason}"))

    async def _on_job_completed(self, job_id: str, text: str, job: Any) -> None:
        """Auto-deliver one settled model-owned job back into the conversation.

        Only when the session is IDLE: a running turn already owns the
        conversation (its model either waited or can 'jobs'), and
        steering-injecting a result nobody asked for mid-turn is noise. Only
        model-registered job types (task, backgrounded bash): host jobs keep
        their own delivery. A job the wait tool already returned is marked
        consumed and stays quiet, or the same result would arrive twice.
        """
        if self._disposed or job is None:
            return
        # Top-level only: a child session is a one-shot runner. Auto-starting
        # an invisible re-entrant child turn from a nested job conflicts with
        # its teardown (and nobody has a panel for it); the child can still
        # consume nested work deliberately through its own jobs/wait tools.
        if self._job_id is not None:
            return
        if getattr(job, "consumed", False) or job.type not in ("task", "bash"):
            return
        if self._is_streaming:
            return
        label = getattr(job, "label", job_id)
        status = getattr(job, "status", "completed")
        summary = (text or "").strip()
        if len(summary) > 2000:
            summary = summary[:2000] + "…[truncated; full result via jobs/wait]"
        delivery = (
            f"background job '{label}' {status}:\n{summary}"
            if summary
            else f"background job '{label}' {status}."
        )
        message = CustomMessage(
            custom_type=JOB_RESULT_MESSAGE_TYPE,
            attribution="user",
            details={"job_id": job_id, "text": delivery},
        )
        self._spawn_background(self._prompt_messages([message]))

    async def _reject_steering(self, command_id: str, reason: str) -> None:
        """Terminally reject one accepted-but-undurable producer steer."""
        for handler in list(self._steering_rejection_handlers):
            try:
                handler(command_id, reason)
            except Exception:  # noqa: BLE001 - one owner cannot hide rejection from others
                logger.exception("steering rejection handler failed")
        await self._emit(
            NoticeEvent(
                text=(
                    f"steering command {command_id} was not saved: {reason}; "
                    "retry with the same command ID"
                )
            )
        )
        # Mobile projections optimistically count accepted steers. Rejection is
        # another terminal exit from that queue, even though it was not delivered.
        await self._emit(SteeringDeliveredEvent(count=1))

    async def _drain_steering(self) -> list[AgentMessage]:
        """Consume the steering queue. Steering messages are real injected
        turns, so they are persisted here — the loop never returns them in its
        ``new_messages``.

        Emits :class:`SteeringDeliveredEvent` when it actually takes something.
        This is the one place that knows a queued message stopped being queued:
        the loop calls it at the boundary where the messages join the context,
        so a front end showing "queued — sends when this step finishes" can
        settle that row to a delivered one instead of leaving the promise up for
        the rest of the session. Silent when the queue is empty, which is the
        overwhelmingly common case — this is called at EVERY tool and message
        boundary, and an event per boundary would be noise with no receiver.
        """
        # Journal notices parked by ``_append_or_park_journal`` rejoin the live
        # context HERE, appended directly rather than returned. This is the
        # loop's own injection boundary — it runs only after ``_append_results``
        # has closed the tool batch — so it is the earliest point at which a
        # notice can rejoin legally, and it lands ahead of any steering message
        # the caller is about to drain in (``_drain_pending`` appends those
        # after), which keeps "you were switched" before the steer the user
        # typed after switching.
        #
        # NOT returned, and that distinction is load-bearing: anything this
        # method hands back becomes a loop INJECTION, and a non-empty injection
        # list keeps the inner loop running (``while has_more_tool_calls or
        # pending``). Returning an advisory notice would therefore buy an entire
        # extra provider call to tell the model something the next turn would
        # have carried for free. Already persisted at park time, so there is no
        # transcript write here either.
        self._flush_context_journal()
        messages: list[AgentMessage] = []
        while not self._steering_queue.empty():
            message = self._steering_queue.get_nowait()
            producer_command_id = self._steering_producers.get(id(message))
            try:
                await self._transcript.append_message(
                    message,
                    producer_command_id=producer_command_id,
                )
            except OSError as exc:
                if producer_command_id is not None:
                    await self._reject_steering(producer_command_id, str(exc))
                else:
                    logger.warning("could not journal steering message", exc_info=True)
                continue
            finally:
                self._steering_producers.pop(id(message), None)
            messages.append(message)
        # The drain is the ONLY consumer, so every courtesy message queued
        # before this boundary just left with it; anything queued after is a
        # fresh count.
        self._courtesy_wake_count = 0
        if messages:
            # After persistence, not before: the receipt says the message is in
            # the conversation, and it is only in the conversation once it is on
            # disk and in the list being handed back to the loop.
            await self._emit(SteeringDeliveredEvent(count=len(messages)))
            # Announce each delivered steer as a user MessageStartEvent too, so
            # EVERY front end paints the steer TEXT — not only a count. A steer
            # injected from the phone during a wake/continuation turn never
            # passed through prompt()'s _run_turn announcement, so the TUI had
            # no user row for it at all. Same event, same de-dup as a prompt:
            # the steering front end's optimistic echo is matched by content.
            for message in messages:
                if isinstance(message, Message) and message.role == "user":
                    await self._emit(MessageStartEvent(message=message))
            self.refresh_frontend_state()
        return messages

    def _has_urgent_steering(self) -> bool:
        """The interrupt poll's peek: True only when queued steering may cancel
        a RUNNING tool. Wakes queued while a turn streamed are counted as
        courtesy (``_courtesy_wake_count``) and excluded — they wait for the
        next successful tool boundary, delivered by ``_drain_steering`` like
        any other steering. A queue holding nothing but courtesy wakes answers
        False, so their timers can never kill an in-flight `bash`."""
        return self._steering_queue.qsize() > self._courtesy_wake_count

    async def _drain_asides(self) -> list[Aside]:
        """Drain queued aside thunks (the loop materializes them at the
        injection boundary and fires commit/discard hooks)."""
        thunks = self._aside_thunks
        self._aside_thunks = []
        return list(thunks)

    def queue_aside(self, thunk: Callable[[], AsideResult]) -> None:
        """Queue a lazy aside message for the next injection boundary. The thunk
        is wrapped so a materialized (non-None, non-stale) message is
        persisted exactly once, at the moment it actually reaches the model.
        A :class:`StaleAside` result passes through unpersisted; the loop
        fires its ``on_discard``."""

        def _wrapped() -> AsideResult:
            message = thunk()
            if message is not None and not isinstance(message, StaleAside):
                self._spawn_background(self._transcript.append_message(message))
            return message

        self._aside_thunks.append(_wrapped)
        # Every aside is a ``hub`` message for this session's model (a child's
        # unprompted "I am blocked" to its parent, or a parent's note/question
        # to a child), and the injection boundary that materializes it is
        # exactly where a parked `wait` returns to. Without this mark the
        # note sits in the thunk list until the wait's budget expires, and a
        # child that speaks up early — which its system prompt tells it to —
        # goes unheard for up to an hour. AFTER the append: the woken tool
        # returns into _drain_asides, which must find the thunk already there.
        # A thunk that later withdraws itself (StaleAside) still woke the
        # wait, which is harmless — the still-running payload just gets
        # re-issued — and cheaper than teaching the tool about withdrawal.
        self._peer_arrival.mark(HUB_MESSAGE_TYPE)

    async def _todo_continuation(self) -> list[AgentMessage]:
        """The loop's follow-up hook: re-assert open todos at the yield boundary.

        The loop ends a turn the instant the model returns no tool calls, even
        with every item still open — a steering message answered in prose was
        enough to end a turn reading ``Todos · 0/6``. A non-empty return here
        sets ``has_more_tool_calls`` and re-enters the outer loop
        (harness/loop.py, ``_collect_yield_injections``), so the model gets
        another turn with its own list in front of it.

        Fires only while the list is MOVING. A model that yields twice with a
        byte-identical list is telling you it cannot proceed — usually it needs
        a decision only the user can make — and nudging it again would burn the
        loop's ``max_paused_turn_continuations`` budget (default 8), end the turn
        with a continuation-limit warning notice, and delay the user's answer by
        up to eight model calls. Any progress earns another nudge; a fresh user
        turn re-arms the latch (see ``_run_turn_pipeline``).

        The nudge it returns is a point-in-time assertion and stops being sent
        the moment the list moves — see :meth:`_live_todo_reminders`, which
        expires it against the fingerprint stamped in its details.

        Invisible to the user by construction, which is exactly why the nudge is
        a ``CustomMessage`` and not a ``Message.user(...)``: follow-ups are
        appended by ``AgentLoop._drain_pending``, which emits no AgentEvent;
        they never enter the run's ``new_messages``, so nothing persists them;
        and the TUI's resume replay (``_render_resumed_history``) branches on
        ``getattr(message, "role", None)``, which a CustomMessage does not have.
        A plain user message would fail all three and would print in the
        transcript as if the user had typed it.
        """
        pending = open_todos(self._session_id)
        if not pending:
            return []
        fingerprint = todo_fingerprint(self._session_id)
        if fingerprint == self._todo_reminder_fingerprint:
            return []
        self._todo_reminder_fingerprint = fingerprint
        return [
            CustomMessage(
                custom_type=TODO_REMINDER_MESSAGE_TYPE,
                attribution="system",
                # The fingerprint rides ALONG with the text: it is what the
                # render path compares to decide the assertion is still true
                # (see :meth:`_live_todo_reminders`).
                details={"text": _todo_reminder_text(pending), "fingerprint": fingerprint},
            )
        ]

    # -- compaction ------------------------------------------------------------

    async def _journal_prunes(
        self, llm_history: Sequence[AgentMessage], pruned_before: set[str]
    ) -> None:
        """Persist the blanking that ``prune_tool_outputs`` just did in memory.

        Pruning is the one place where the live context and the transcript
        drift apart: the session throws away a multi-kilobyte tool output,
        the transcript keeps it, and the next resume replays the original
        back into the prompt — so a resumed session costs MORE per turn than
        the session it resumed. Journalling closes that gap and, once the
        dead bytes are worth a rewrite, ``compact_file`` takes them off disk
        too. Best-effort: a transcript write that fails must never abort the
        turn's compaction, which is load-bearing where this is an optimisation.
        """
        newly = [
            message
            for message in llm_history
            if isinstance(message, Message)
            and message.id not in pruned_before
            and (message.provider_payload or {}).get("pruned")
        ]
        if not newly:
            return
        try:
            for message in newly:
                await self._transcript.append_prune(message.id, message.text)
            await self._transcript.compact_file()
        except OSError as exc:
            logger.warning("could not journal pruned tool outputs: %s", exc)

    async def _persist_progress(self, messages: Sequence[AgentMessage]) -> None:
        """Flush whatever the running turn has produced so far, best-effort.

        The durability floor for a turn. :meth:`_persist_new_messages` is the
        mechanism and stays strict (its callers on the normal path want a
        failure to surface); this wrapper is for the two places that must
        never fail the turn they are protecting — the tool-loop boundary hook
        and ``_run_turn``'s ``finally`` — where the alternative to a swallowed
        write error is losing the whole run instead of one message.

        Snapshotted with ``list()`` because the live context is mutated by the
        loop and the ``finally`` caller passes ``self._context.messages``
        itself.
        """
        # Deliberately NOT gated on ``self._disposed``. ``dispose()`` sets that
        # flag BEFORE it aborts and awaits the in-flight turn, precisely so
        # that turn's persistence "must land on a live transcript" — so a
        # ``_disposed`` guard here would suppress the one flush dispose is
        # waiting for.
        #
        # Writing is safe at any point in teardown, and for a stronger reason
        # than "the transcript is still open": there is no open handle to lose.
        # ``Transcript`` has no ``close()``; ``flush()`` is an explicit no-op
        # ("writes are flushed per append"), and ``_append`` opens, writes and
        # closes the file per call. So a disposing — or already disposed —
        # session can still append, which also means the late flush from a turn
        # that outlives dispose's 5s shielded wait still lands (review round 2,
        # R6: the earlier wording here described a lifecycle that does not
        # exist, and would send the next reader hunting for a close to race).
        try:
            await self._persist_new_messages(_paired_prefix(messages))
        except asyncio.CancelledError:
            # Cancellation must propagate: swallowing it here would keep a
            # turn alive that the session is trying to tear down.
            raise
        except Exception:  # noqa: BLE001 — durability is best-effort
            logger.warning("could not persist turn progress", exc_info=True)

    async def _persist_new_messages(self, messages: Sequence[AgentMessage]) -> None:
        """Append every message not already in the transcript, in order.

        Idempotent by message id, which is what lets the mid-turn compaction
        gate flush a run's messages early (it needs a persisted cut target)
        without the post-run persistence pass writing them a second time. The
        transcript stores a message under its OWN id, so "already stored" is
        an exact check rather than a heuristic.
        """
        stored = {entry.id for entry in self._transcript.entries()}
        for message in messages:
            if not _is_persistable_message(message):
                continue
            if getattr(message, "id", None) in stored:
                continue
            await self._transcript.append_message(message)

    async def _on_turn_end(self, messages: list[AgentMessage]) -> list[AgentMessage] | None:
        """Mid-turn compaction gate — runs INSIDE the tool loop, at the safe
        boundary after each tool batch lands and before the next model call.

        Without this, a single long tool run (dozens of calls, no user turn)
        grows past the window with no relief until the run ends — the next
        provider request then fails the whole run, and the post-turn pass
        recovers a turn that never needed to break. The knob
        (``values.compaction.mid_turn_enabled``) defaults on; the pass itself
        is the same single plan+commit the post-turn gate and ``/compact``
        share, so there is no second compaction implementation to drift.

        Returns the replacement context so the loop can prune its run
        accumulator (see ``LoopConfig.on_turn_end``), or ``None`` when no
        compaction ran. The replayability guard in ``_plan_compaction``
        (kept[0] must be a transcript entry) naturally constrains mid-run
        cuts to already-persisted history, which is where a cut belongs
        mid-run anyway.
        """
        if self._disposed:
            return None
        # Durability FIRST, unconditionally, and before any compaction
        # decision: this hook is the only place that sees the run's messages
        # at a safe boundary, and every early return below it used to leave
        # the whole run unpersisted until the run ended. Mid-run persistence
        # existed only as a SIDE EFFECT of the compaction pass below (it needs
        # a persisted cut target), so with ``mid_turn_enabled`` off — or
        # simply below the threshold — a long tool run kept 100% of its work
        # in memory. A session killed there (crash, SIGKILL, Ctrl+C) replayed
        # to nothing but the user's prompt. Measured on a 6-step run: one
        # entry on disk without this, ten with it.
        #
        # Idempotent by message id, so the post-run pass still writes each
        # message exactly once.
        await self._persist_progress(messages)
        # The fork drains HERE, immediately after that persist and before the
        # compaction decision below. Order is the whole argument: the clone is a
        # file copy, so it needs the run's messages already on disk (the line
        # above) and a transcript whose tail is not an unanswered ``tool_use``
        # (this boundary is after the batch's results landed). Draining before
        # any compaction also means the fork inherits the transcript the user
        # was looking at when they typed the command, rather than a rewritten
        # head they never saw.
        await self._drain_pending_fork()
        # The post-run usage scan has not happened yet — the boundary
        # snapshot carries the assistant message that just finished, whose
        # usage is the provider's ground truth for the trigger math below AND
        # for the prompt-cache TTL hint a mid-turn aside or advisor call is
        # stamped with. Unconditional, ahead of the compaction gate: the hint
        # must track the conversation's size even for a host that switched
        # mid-turn compaction off.
        self._note_usage(messages)
        try:
            from local_operator.compaction.api import CompactionSettings
        except ImportError:
            return None
        settings = self._compaction_settings or CompactionSettings()
        # getattr: hosts (and the test suite) may inject a duck-typed
        # settings model that predates the knob; absence means the §C default
        # (on), same posture as _resolve_strategy's capability probes.
        if not settings.enabled or not getattr(settings, "mid_turn_enabled", True):
            return None
        # Cheap pre-gate: when the provider just reported its context size and
        # that figure already fails the trigger, skip the full plan — the
        # plan renders the whole history to prove the same thing, and this
        # hook fires at EVERY continuing tool-loop boundary.
        # ``should_compact`` resolves the threshold itself (percent x window
        # vs the absolute ceiling, ONE resolver), so this gate holds no
        # threshold arithmetic of its own to drift from the plan's; a mirrored
        # copy of that math here is what once let a session's gate and its
        # receipt disagree about when a pass was due. No provider figure
        # (usage not yet reported) falls through to the plan, whose own
        # upper-bound proof is the cheap path there.
        provider_reported = (
            self._last_usage.context_tokens if self._last_usage is not None else None
        )
        # Off-loop advisory (BETA, inert by default) — spawned ABOVE the cheap
        # pre-gate, and the ordering is the whole point of the feature.
        #
        # The advisor's entire purpose is to pull the trigger EARLIER, so its
        # operating band is by definition BELOW the ordinary trigger. The
        # pre-gate below returns as soon as the provider figure fails that
        # ordinary trigger, so a spawn placed after it could only ever be
        # reached once the context had already passed the line — the one
        # moment a size pass was about to fire anyway and the advice is worth
        # nothing. Measured before this was moved: ctx 350k/400k/500k/590k on
        # a 600k trigger produced ZERO advisor calls, and only 650k produced
        # one. The feature was inert across the whole 200k-600k band it exists
        # for (agent review round 1, blocker-1).
        #
        # Spawning first is safe and cheap: ``_maybe_spawn_advisor`` applies
        # its own gates (beta flag, advisor_trigger_tokens, in-flight latch,
        # call ceiling, cooldown, kill switch) and returns immediately for a
        # session that has not opted in, so a default session pays one
        # attribute read. Nothing is awaited, so the boundary is not slowed.
        #
        # It also deliberately does NOT depend on ``_persist_new_messages``
        # below: the advisor reads the live context to form an opinion and
        # writes nothing, so it has no replayable-cut requirement of its own.
        #
        # A finished BACKGROUND pass is applied here, above the pre-gate, for
        # exactly the reason the spawn sits above it: an advisor-triggered pass
        # fires below the ordinary trigger, so the pre-gate returns before it
        # every time and the pass would wait forever for a boundary that never
        # comes. Applied BEFORE the spawn so the advisor forms its opinion
        # about the context the pass just produced rather than the one it
        # replaced. ``_persist_progress`` at the top of this hook has already
        # run, so the kept window is on disk.
        applied = await self._apply_pending_compaction() is not None
        self._maybe_spawn_advisor()
        if provider_reported is not None:
            from local_operator.compaction import api as compaction_api

            # The pre-gate must ask the SAME question the plan gate will, or it
            # answers "no pass due" for a context the plan gate would have
            # compacted on advice — the hint would be spawned, land, and then
            # be unreachable because this gate returned first. It PEEKS at the
            # pending hint rather than consuming it: the hint is consumed at
            # exactly one point (``_plan_compaction``), and a pre-gate that
            # consumed it would leave the plan gate nothing to act on.
            if not _should_compact(
                compaction_api,
                provider_reported,
                self.effective_model.context_window,
                settings,
                self._has_pending_advisory(settings),
            ):
                # A background pass that just landed still has to reach the
                # loop, or the run accumulator keeps the history the pass
                # removed and the next request re-sends every token of it.
                return list(self._context.messages) if applied else None
        # Persist what the run has produced SO FAR before planning a cut.
        #
        # The post-run path writes the whole run at once (see the persistence
        # block in ``prompt``), which is too late for a mid-run pass: the cut
        # has to land on an already-persisted entry or ``_plan_compaction``
        # refuses it as ``cut_not_replayable``, and mid-run the tail of the
        # history is exactly the part the run just made. That refusal fired at
        # every boundary of a long tool run, so the gate correctly decided to
        # compact and was then blocked from doing it, and the context kept
        # growing until the run ended.
        #
        # Appending early is safe because the transcript is append-only and
        # keyed by message id: the post-run loop re-appends nothing that is
        # already stored (``_persist_new_messages`` skips known ids), and a
        # crash mid-run leaves a transcript that replays to what actually
        # happened rather than losing the run outright.
        await self._persist_new_messages(messages)
        planned = await self._plan_compaction(respect_threshold=True)
        if isinstance(planned, CompactionOutcome):
            # No new pass is due. Still hand the loop the rebuilt context when
            # a background pass just landed, or the run accumulator keeps the
            # history the pass removed and the next request re-sends it.
            return list(self._context.messages) if applied else None
        if planned.advisor_hint is not None and self._pass_may_run_off_the_turn(planned):
            # An EARLY pass, on advice, with the context still below the
            # ordinary trigger: nothing is forcing relief, so it runs off the
            # turn and applies at a later boundary. The context is unchanged
            # right now, so the loop continues on the history it already holds.
            #
            # Both conditions are load-bearing. A hint alone is NOT enough to
            # defer (round 4, MAJOR-1): a hint can be in hand while the context
            # is genuinely over the ceiling, and deferring there is exactly the
            # safety net becoming asynchronous. ``_pass_may_run_off_the_turn``
            # asks the resolved trigger itself, so an over-ceiling context
            # falls through to the synchronous pass below.
            #
            # A spawn REFUSED (one already in flight) skips the pass rather
            # than falling back to the inline one. Below the trigger nothing is
            # forcing relief, so making the user wait for a summarization call
            # would reintroduce exactly the stall this path removes, at the
            # worst moment: a boundary that already has a pass running. The
            # hint was consumed, and the advisor produces another when it next
            # runs.
            self._spawn_compaction_pass(planned, reason="mid-turn")
            return list(self._context.messages) if applied else None
        outcome = await self._run_compaction(planned, reason="mid-turn")
        if not outcome.ran:
            return list(self._context.messages) if applied else None
        self._settle_advisor(planned, outcome)
        return list(self._context.messages)

    async def _maybe_compact(self) -> None:
        """Post-turn compaction check — the AUTOMATIC trigger.

        Everything except the trigger itself is shared with the manual one
        (:meth:`compact_now`): :meth:`_plan_compaction` decides, and
        :meth:`_run_compaction` commits. Order (binding orchestrator
        decisions):

        1. ``prune_tool_outputs`` over the LLM history (in-place blanking of
           superseded/useless tool outputs) BEFORE the trigger math.
        2. Trigger on ``compaction_context_tokens`` (max of provider-reported
           context size and the local estimate).
        3. Threshold: whatever ``compaction.thresholds.resolve_threshold_tokens``
           resolves for this window — ``min(threshold_percent * context_window,
           threshold_tokens)``, defaults 80% and 600k. The gate never derives
           it here; a mirrored formula in the session is how a 1M-context
           session ended up compacting at ~235k.
        4. Strategy resolution: snapcompact for vision models (archive stored
           under ``preserve_data['snapcompact']``), context-full otherwise;
           any snapcompact failure falls back to LLM summarization.
        5. After a successful pass, schedule auto-continue only when the
           residual cleared the recovery band (``residual <= 0.8 * threshold``).
        """
        # The turn is over and the next one has not started: the safest
        # boundary there is, and the one a mid-turn spawn most often lands on
        # (a pass spawned at the last tool batch finishes while the model
        # writes its closing message).
        landed = await self._apply_pending_compaction()
        if landed is not None:
            # A background pass just rebuilt the context. It owes the user the
            # same post-pass bookkeeping a synchronous one does, or an early
            # advisory pass would silently lose the auto-continuation that
            # resumes an interrupted task.
            self._after_compaction_pass(*landed)
            return
        planned = await self._plan_compaction(respect_threshold=True)
        if isinstance(planned, CompactionOutcome):
            # Refused: below threshold, disabled, nothing worth summarizing.
            # The automatic path has nobody to tell — a turn that did not need
            # compacting must not narrate that fact every time.
            return
        if planned.advisor_hint is not None and self._pass_may_run_off_the_turn(planned):
            # Advisory AND genuinely below the configured trigger: nothing is
            # forcing relief right now and the next turn can safely start on
            # the uncompacted context. The auto-continue bookkeeping below
            # belongs to the pass and runs when it applies. A refused spawn
            # skips the pass for the reason the mid-turn gate spells out: an
            # inline fallback here is the stall this path exists to remove.
            #
            # A hint in hand does NOT by itself mean the context is below the
            # trigger (round 4, MAJOR-1); when it is not, this falls through to
            # the synchronous pass, which is the safety net and must stay so.
            self._spawn_compaction_pass(planned, reason="context-window")
            return
        outcome = await self._run_compaction(planned, reason="context-window")
        if not outcome.ran:
            return
        self._settle_advisor(planned, outcome)
        self._after_compaction_pass(planned, outcome)

    def _after_compaction_pass(self, plan: _CompactionPlan, outcome: CompactionOutcome) -> None:
        """Post-turn bookkeeping owed by ANY pass that lands at the turn edge.

        Shared by the synchronous gate and by a background pass applied there,
        so the two cannot drift about what a completed pass implies. Called
        only after ``_settle_advisor``, which owns the anti-thrash side.
        """
        # The provider usage in `_held_end.messages` remains authoritative for
        # BILLING, but its occupancy predates this pass. If it is allowed to win
        # the later agent-end reduction, every frontend briefly paints `after`
        # from compaction_end and then rebounds to `before` at turn settlement.
        self._held_context_tokens = outcome.tokens_after

        # (5) Recovery band: only schedule a continuation when the pass
        # actually created headroom (an anti-thrash guard).
        if getattr(plan.settings, "auto_continue", False):
            compaction_api = plan.compaction_api
            threshold = compaction_api.resolve_threshold_tokens(
                self.effective_model.context_window, plan.settings
            )
            if outcome.tokens_after <= compaction_api.RECOVERY_BAND * threshold:
                self._continuation_queue.append(Message.user(_CONTINUATION_PROMPT))

    async def compact_now(self) -> CompactionOutcome:
        """Compact the context NOW, on the user's explicit request (``/compact``).

        The manual trigger runs THE SAME PASS as the automatic one — one
        :meth:`_plan_compaction`, one :meth:`_run_compaction`, one strategy
        resolver, one pair of ``compaction_start``/``compaction_end`` events —
        with the threshold gate skipped, because the user asking IS the
        trigger. A second entry point would be free to drift: the manual pass
        would keep summarizing with a text model long after the automatic one
        had moved a vision model onto snapcompact.

        Every state a manual trigger can be pressed in that the automatic gate
        never sees comes back as a REFUSAL with a reason, never as silence: the
        bug this method fixes was a ``/compact`` that changed nothing, and a
        refusal nobody can see is the same frame.
        """
        if self._compacting:
            return CompactionOutcome(
                ran=False,
                reason="already_running",
                detail="a compaction is already running",
            )
        if self._is_streaming or self._turn_lock.locked():
            # A running turn owns the message list — the loop holds it across
            # tool batches — and rebuilding it under the loop is how a tool
            # call loses the result it is waiting for. The turn LOCK is
            # consulted as well as the streaming flag for the reason
            # :meth:`adopt_aside` spells out: the flag covers ``_run_turn``
            # alone, while the lock is held across the whole pipeline including
            # a post-compaction auto-continuation, and the gap between them is
            # a window a manual pass must not splice into.
            return CompactionOutcome(
                ran=False,
                reason="turn_running",
                detail=(
                    "a turn is still running — compaction rewrites the history the turn is "
                    "holding, so it has to wait for the turn to finish"
                ),
            )
        # HELD for the whole pass, exactly as a turn holds it: the pass replaces
        # ``_context.messages``, and a prompt or a wake delivery that started in
        # the middle would build its request from a history being rewritten
        # underneath it. `_compacting` is what lets `prompt` say WHY it is
        # refusing while this is in flight.
        await self._turn_lock.acquire()
        self._compacting = True
        try:
            planned = await self._plan_compaction(respect_threshold=False)
            if isinstance(planned, CompactionOutcome):
                return planned
            return await self._run_compaction(planned, reason="manual")
        finally:
            self._compacting = False
            # `record_shell` queues while manual compaction owns the message
            # list. Fold those receipts into the newly compacted context before
            # another prompt can acquire the lock and build its request.
            await self._flush_shell_records()
            self._turn_lock.release()

    @staticmethod
    async def _offloaded(compaction_api: Any, name: str, *args: Any) -> Any:
        """Call one compaction ruler off the event loop when it is worth it.

        The rulers (``estimate_messages_tokens``, ``find_cut_point``) tokenize
        the whole history and run on EVERY turn. One event loop serves the
        parent session, every subagent and the TUI repaint, so counting inline
        made one agent's threshold check stall all of them — measured at up to
        860 ms with eight children running, with 116 of 121 stall samples
        inside the encoder. tiktoken's ``encode`` releases the GIL, so a worker
        thread converts that stall into real parallelism (measured: 90 ms of
        loop stall becomes 0.7 ms, and the same work finishes ~3x sooner).

        Resolved by NAME off the passed module rather than closed over at
        import: the module is looked up per call (and tests substitute partial
        doubles for it), so binding the function early would call the real
        ruler while a test believed it had pinned one.

        Small histories stay inline because the thread hop costs more than the
        encode it saves, and a module that does not expose ``history_chars``
        (a partial test double) is treated the same way — degrading to the
        inline path is always correct, just slower, and must never be a crash.
        """
        func = getattr(compaction_api, name)
        probe = getattr(compaction_api, "history_chars", None)
        threshold = getattr(compaction_api, "OFFLOAD_MIN_CHARS", None)
        if not callable(probe) or not isinstance(threshold, int):
            return func(*args)
        # ``args[0]`` is the history for both rulers; anything after it is a
        # scalar setting, so sizing on the first argument is sufficient. The
        # probe comes off a module resolved at runtime, so its return type is
        # unknown here: a double that returns a non-number takes the inline
        # path rather than raising, same as one that omits the probe entirely.
        size = probe(args[0])
        if not isinstance(size, int) or size < threshold:
            return func(*args)
        # Snapshot the sequence: the worker must never walk a list the loop
        # could mutate underneath it (pruning mutates histories in place).
        snapshot = list(args[0])
        rest = args[1:]
        return await asyncio.to_thread(lambda: func(snapshot, *rest))

    async def _plan_compaction(
        self, *, respect_threshold: bool
    ) -> _CompactionPlan | CompactionOutcome:
        """Everything decided BEFORE a pass commits, for both triggers.

        Returns a plan when a compaction should run, or the
        :class:`CompactionOutcome` explaining why it must not. ``respect_threshold``
        is the ONLY difference between the two callers: the automatic gate fires
        on the context size, the manual one fires because it was asked.

        Side effects happen here on purpose: pruning (in-place blanking of
        superseded tool outputs) runs before the trigger math, so a context the
        prune alone brought back under the line never buys a summary. It also
        means a manual ``/compact`` on a context with nothing to summarize has
        still reclaimed whatever the prune found.
        """
        try:
            from local_operator.compaction import api as compaction_api
        except ImportError:
            return CompactionOutcome(
                ran=False,
                reason="unavailable",
                detail="compaction is not available in this build",
            )

        settings = self._compaction_settings or compaction_api.CompactionSettings()
        strategy = self._resolve_strategy(settings)
        if not settings.enabled or strategy == "off":
            return CompactionOutcome(
                ran=False,
                reason="disabled",
                detail="compaction is switched off in config (values.compaction)",
            )
        # NOTE: the threshold is NOT derived here. ``should_compact`` below
        # resolves ``min(threshold_percent * window, threshold_tokens)`` through
        # the one resolver in compaction.thresholds, and ``_maybe_compact``'s
        # recovery band asks the same resolver for the same number. This block
        # used to pre-bake a threshold into ``settings.threshold_tokens``,
        # which turned a second knob (a defensive ceiling) into a second
        # definition of "when to compact" — and a 1M-context session firing at
        # ~235k, 23% of its window, was the result.

        llm_history = self._render_for_compaction()
        if not llm_history:
            return CompactionOutcome(
                ran=False,
                reason="nothing_to_compact",
                detail="the conversation is empty — there is nothing to compact",
            )

        # (1) Prune before deciding: blanked outputs shrink the estimate and
        # may avoid a compaction pass entirely. Mutates messages in place.
        # The idle flush compares against the last PROVIDER request, not turn
        # bookkeeping, so a genuinely idle session reclaims the warm region.
        now_ms = int(time.time() * 1000)
        pruned_before = _pruned_ids(llm_history)
        try:
            compaction_api.prune_tool_outputs(llm_history, now_ms, self._last_provider_request_ms)
        except (ImportError, AttributeError):
            pass  # optional pruning hook absent; degrade to no pruning
        await self._journal_prunes(llm_history, pruned_before)

        # (2) Trigger math: prefer the provider's ground-truth context size.
        provider_reported = (
            self._last_usage.context_tokens if self._last_usage is not None else None
        )

        # (2a) The advisory, consumed at THIS single point and nowhere else.
        # It becomes exactly two things: the ``advisory_ok`` flag handed to
        # ``should_compact`` and the effective keep-recent handed to
        # ``find_cut_point``. ``_run_compaction`` is untouched and manual
        # ``/compact`` never gets here with a hint, so there is no path by
        # which a hallucinated hint reaches the summarizer or the transcript.
        # ``_take_advisor_hint`` returns only a USABLE hint (fresh, advisor
        # still enabled, and actually saying "now"), so a hint in hand IS the
        # authorisation — the peek in the mid-turn pre-gate applies the same
        # rule, which is what keeps the two gates from disagreeing.
        advisor_hint = self._take_advisor_hint(settings) if respect_threshold else None
        advisory_ok = advisor_hint is not None

        if respect_threshold:
            # Cheap proof first. ``should_compact`` is strictly monotonic in
            # context_tokens and ``compaction_context_tokens`` is monotonic in
            # the local estimate, so a rigorous UPPER bound that already fails
            # the threshold test proves the exact estimate fails it too — same
            # early return, same observable behaviour. This matters because the
            # first exact estimate in a process loads tiktoken's cl100k_base
            # table (~84 ms, ~43.6 MB RSS, measured with
            # scripts/bench_base_overhead.py), and compaction runs on EVERY
            # turn while the typical session never comes near its threshold —
            # so every short run was buying a 43.6 MB tokenizer to be told
            # "no". A MANUAL pass is going to load it anyway, so it skips
            # straight to the exact figure it has to report.
            bound = compaction_api.messages_tokens_upper_bound(llm_history)
            if not _should_compact(
                compaction_api,
                compaction_api.compaction_context_tokens(provider_reported, bound),
                self.effective_model.context_window,
                settings,
                advisory_ok,
            ):
                return CompactionOutcome(ran=False, reason="below_threshold")

        # Both of the next two calls tokenize the whole history, and both used
        # to run inline on the event loop EVERY turn. Because one loop serves
        # the parent session, every subagent and the TUI repaint, that made one
        # agent's threshold check a global stall: a stall trace of eight
        # concurrent subagents put 116 of 121 blocking samples inside the
        # encoder reached from here (worst single stall 860 ms).
        # ``_offloaded`` hands large histories to a worker thread, where
        # tiktoken's GIL release lets them run genuinely in parallel; small
        # ones still run inline so a short session pays no thread-hop tax.
        local_estimate = await self._offloaded(
            compaction_api, "estimate_messages_tokens", llm_history
        )
        context_tokens = compaction_api.compaction_context_tokens(provider_reported, local_estimate)
        if respect_threshold and not _should_compact(
            compaction_api,
            context_tokens,
            self.effective_model.context_window,
            settings,
            advisory_ok,
        ):
            return CompactionOutcome(ran=False, reason="below_threshold")

        # (2b) Task-aware preserve window. The cut is recency-shaped
        # (``keep_recent_tokens``) while the session is task-shaped, and on the
        # measured session that mismatch cut inside a live task on five of
        # seven passes. ``task_boundary_floor`` is a FLOOR, never a
        # replacement, so this can only keep MORE history than the recency
        # rule asked for — which is what makes an earlier trigger safe. An
        # accepted advisory may widen it further, never narrow it (the
        # validator rejects a narrowing hint outright).
        #
        # This changes the CUT, not the TRIGGER: it runs for every pass,
        # advisory or not, and introduces no second gate.
        keep_recent = settings.keep_recent_tokens
        boundary_floor: Any = getattr(compaction_api, "task_boundary_floor", None)
        if callable(boundary_floor):
            # Resolved by name off the module for the tolerance ``_offloaded``
            # grants its rulers: a partial test double that predates this
            # degrades to plain recency rather than crashing the pass.
            genuine_user_ids = {
                message.id
                for message in self._context.messages
                if isinstance(message, Message) and message.role == "user"
            }
            floor_value: Any = boundary_floor(
                llm_history,
                genuine_user_ids,
                cap=self._advisor_floor_cap(settings),
            )
            keep_recent = max(keep_recent, int(floor_value))
        if advisor_hint is not None:
            # Clamped with the SAME cap the local floor uses. Uncapped, a wide
            # but perfectly legal hint turns "keep more" into "never compact":
            # the widen-only guard has no upper bound of its own, and
            # ``_candidate_messages`` offers text messages while
            # ``preserve_tokens`` sums every intervening entry, so in a
            # tool-heavy session the widest legal anchor spans far more context
            # than the candidate list suggests. Measured at 2.7x the cap
            # (800,340 tokens against the 300,000 cap of the time), which was
            # enough to turn a mandatory 800k-on-1M pass into
            # ``nothing_to_compact`` — the exact failure the cap exists to
            # prevent, arrived at through the guard meant to make hints safe
            # (agent review round 1, major-1).
            #
            # That cap is now ``min(keep_recent * _TASK_FLOOR_KEEP_MULTIPLE,
            # threshold // 2)``. On a large window the task term binds and the
            # clamp TIGHTENS (100,000 against the 300,000 of the incident), so
            # the same hint is bounded harder than it was when the incident was
            # recorded. Below the ~250k crossover the capacity term binds and
            # the clamp is exactly what shipped. Either way it never loosens —
            # which is the property this comment needs, and it is stated per
            # window because the tightening is NOT universal (agent review
            # round 1, blocker-1: an earlier draft dropped the capacity term
            # and did loosen it on 65% of the registry).
            #
            # The clamp cannot narrow below the local floor: ``max`` is applied
            # against ``keep_recent``, which already carries the capped
            # ``task_boundary_floor``, so the widen-only property survives.
            capped_hint = min(int(advisor_hint.preserve_tokens), self._advisor_floor_cap(settings))
            keep_recent = max(keep_recent, capped_hint)

        cut = await self._offloaded(compaction_api, "find_cut_point", llm_history, keep_recent)
        if cut is None or cut <= 0:
            # ``find_cut_point`` is the ONE definition of "worth summarizing":
            # the kept window has to reach ``keep_recent_tokens`` and at least
            # two real messages have to fall outside it. Both states a manual
            # trigger runs into land here — a context too small to have older
            # history, and a context whose older history a previous pass has
            # already summarized — so the refusal names which one it is rather
            # than guessing.
            if local_estimate <= keep_recent:
                detail = (
                    f"nothing to compact: the whole conversation is ~{local_estimate:,} tokens "
                    f"and the most recent {keep_recent:,} are kept verbatim"
                )
            else:
                detail = (
                    "nothing left to compact: everything older than the recent window is "
                    "already summarized"
                )
            return CompactionOutcome(
                ran=False,
                reason="nothing_to_compact",
                detail=detail,
                tokens_before=local_estimate,
                tokens_after=local_estimate,
            )

        # The kept window must start at an entry the transcript can replay:
        # first_kept_entry_id is persisted and matched on resume, and a cut
        # whose first kept message has no transcript entry (a converter-minted
        # id) would make replay drop the whole kept window silently.
        entry_ids = {entry.id for entry in self._transcript.entries()}
        if llm_history[cut].id not in entry_ids:
            logger.warning(
                "compaction cut rejected: kept[0].id %s is not a transcript entry",
                llm_history[cut].id,
            )
            return CompactionOutcome(
                ran=False,
                reason="cut_not_replayable",
                detail=(
                    "the history has no replayable cut point yet — compaction would drop the "
                    "messages it keeps on the next resume"
                ),
                tokens_before=local_estimate,
                tokens_after=local_estimate,
            )

        return _CompactionPlan(
            compaction_api=compaction_api,
            settings=settings,
            strategy=strategy,
            llm_history=llm_history,
            cut=cut,
            context_tokens=context_tokens,
            tokens_before=local_estimate,
            # ATTRIBUTION, not influence. The hint above may have widened the
            # preserve window for this pass whatever triggered it — that is the
            # widen-only guarantee and it applies to every pass. But a pass the
            # context would have fired ANYWAY, on size alone, was not caused by
            # the advice, and ``advisor_hint`` on the plan is read by exactly
            # two consumers that both mean "the advisor caused this":
            #
            # - ``_advisor_detail`` renders "advisor: <reason>" on the receipt.
            #   On a size-triggered pass that tells the user the advisor did
            #   something the ordinary trigger did by itself.
            # - ``_settle_advisor`` JUDGES the advisor by what the pass
            #   reclaimed, and can DISABLE it for the rest of the session. A
            #   size pass counting against the advisor's reclaim record lets an
            #   ordinary ceiling pass switch off a feature it never invoked;
            #   verified reachable in agent review round 5 (MINOR-2), where a
            #   700k pass over a 600k ceiling both claimed advisor credit on
            #   the receipt and set ``_advisor_disabled``.
            #
            # So the hint is still CONSUMED (single-use: it is an opinion about
            # one moment and must not lower a later gate) and still shapes the
            # cut, but it is only carried onto the plan when the advice is what
            # made the pass happen. Same predicate as the async routing, so
            # "advisory pass" means one thing across the file.
            advisor_hint=(
                advisor_hint
                if advisory_ok
                and not self._fires_on_size_alone(compaction_api, context_tokens, settings)
                else None
            ),
        )

    def _settle_advisor(self, plan: _CompactionPlan, outcome: CompactionOutcome) -> None:
        """Anti-thrash bookkeeping after an ADVISOR-triggered pass.

        Two guards, both about the same failure: an advisor that keeps saying
        "now" to a context it cannot actually relieve.

        1. **Cooldown.** After any advisory pass the advisor is suppressed for
           ``advisor_cooldown_turns``. The pass just moved the very boundary
           the advisor would be asked to judge, so an immediate re-ask is a
           question about a conversation that no longer exists.

        2. **Kill switch, non-negotiable.** An advisory pass that RECLAIMED
           almost nothing did not merely fire early, it fired where there was
           nothing to reclaim — the same signature as the live dead-loop bug
           ``RECOVERY_BAND`` was added for, except the advisor would keep
           re-authorising it below the configured threshold. So the advisor is
           DISABLED for the rest of the session and the fact is logged once,
           loudly enough to be found. Nothing re-enables it short of a new
           session: a feature that can spend money in a loop must fail closed.

           The test is a REDUCTION test, and it has to be. It previously
           compared the absolute residual against ``RECOVERY_BAND *
           advisor_floor``, which is independent of what the pass achieved and
           was wrong three separate ways (agent review round 2, major-3):

           - It disabled the advisor after obviously good passes. Measured on
             the real path: ``400,000 -> 242,857`` (39.3% reclaimed) tripped
             it, and that was this feature's own headline evidence.
           - It made ``RECOVERY_BAND`` mean two different things in one file.
             The auto-continue guard below asks ``residual <= 0.8 * threshold``
             (480k on a 1M window); this asked ``residual <= 0.8 * floor``
             (160k), 3.75x stricter, so one residual was simultaneously
             "created headroom, safe to continue" and "reclaimed nothing,
             switch the feature off".
           - It was close to unsatisfiable. A pass may legally preserve up to
             ``_advisor_floor_cap`` (300k on the shipped default), which alone
             exceeds the 160k residual the old rule demanded, so a single long
             task was enough to end the session's advisor.

           ``cleared_headroom(residual, before)`` is the notion the compaction
           module already owns for "how much did this pass actually free", so
           the rule is expressed with it rather than with a second private
           formula. The advisor survives whenever a pass freed at least
           :data:`_ADVISOR_MIN_RECLAIM_FRACTION` of the context it started
           from; a pass that freed less than that genuinely is not helping.

        An ordinary size-triggered pass (``advisor_hint is None``) is not the
        advisor's business and returns immediately.
        """
        hint = plan.advisor_hint
        if hint is None:
            return
        settings = plan.settings
        cooldown = int(getattr(settings, "advisor_cooldown_turns", 0) or 0)
        if cooldown > 0:
            self._advisor_cooldown_until = self._generation + cooldown
        try:
            compaction_api = plan.compaction_api
            # Measured against what the pass STARTED from, which is the figure
            # the gate acted on and the receipt reports as "before".
            before = int(outcome.tokens_before or plan.context_tokens or 0)
            reclaimed: Any = compaction_api.cleared_headroom(int(outcome.tokens_after), before)
        except Exception:  # noqa: BLE001 — a partial double must not break the pass
            return
        if before <= 0:
            # No usable "before" figure: refuse to judge rather than guess. A
            # kill switch that fires on missing data disables the feature for
            # a measurement failure instead of a behavioural one.
            return
        fraction = int(reclaimed) / before
        if fraction >= _ADVISOR_MIN_RECLAIM_FRACTION:
            return
        self._advisor_disabled = True
        logger.warning(
            "compaction advisor disabled for this session: an advisor-triggered pass "
            "reclaimed only %s of %s tokens (%.1f%%, below the %.0f%% floor) — the "
            "advice was not reclaiming headroom",
            f"{int(reclaimed):,}",
            f"{before:,}",
            fraction * 100,
            _ADVISOR_MIN_RECLAIM_FRACTION * 100,
        )

    def _advisory_is_usable(self, hint: Any | None, settings: Any) -> bool:
        """Whether a pending hint may lower the trigger right now.

        ONE definition of "usable", shared by the peek
        (:meth:`_has_pending_advisory`) and the consume
        (:meth:`_take_advisor_hint`). Two copies of this rule is how the
        mid-turn pre-gate and the plan gate would come to disagree about
        whether a pass is due — the same class of drift
        ``resolve_threshold_tokens`` exists to prevent for the threshold
        itself.

        Stale hints are not usable. ``advisor_every_n_turns`` bounds how far
        behind a hint may be: one produced more than an advisory interval ago
        describes a conversation that has since moved, and a late hint is
        nearly no hint.
        """
        if hint is None or self._advisor_disabled:
            return False
        if not getattr(hint, "compact_now", False):
            return False
        every = int(getattr(settings, "advisor_every_n_turns", 0) or 0)
        if every > 0 and self._generation - int(getattr(hint, "turn_index", 0)) > every:
            return False
        return True

    def _has_pending_advisory(self, settings: Any) -> bool:
        """PEEK: is there a usable hint the plan gate would act on?

        Read-only on purpose. The mid-turn pre-gate has to ask the same
        question the plan gate will, or it short-circuits a boundary the plan
        gate would have compacted on advice — but it must not CONSUME the
        hint, because consumption happens at exactly one point and a pre-gate
        that took it would leave the plan gate nothing to act on (the pass
        would then be refused as below_threshold, one gate later).
        """
        return self._advisory_is_usable(self._advisor_hint, settings)

    def _take_advisor_hint(self, settings: Any) -> Any | None:
        """The pending advisory, CONSUMED (single-use) and freshness-checked.

        Single-use because a hint is an opinion about one moment: leaving it
        in place would let one call's judgement lower the trigger on every
        subsequent gate until the next call replaced it, which is a stuck
        threshold wearing an advisory's clothes. The slot is cleared before
        every early return, so an unusable hint cannot persist either.
        """
        hint = self._advisor_hint
        self._advisor_hint = None
        if not self._advisory_is_usable(hint, settings):
            if hint is not None:
                logger.debug("compaction advisor: discarded unusable hint")
            return None
        return hint

    async def _run_compaction(self, plan: _CompactionPlan, *, reason: str) -> CompactionOutcome:
        """Commit one compaction pass — THE pass, for both triggers.

        ``reason`` rides the two events so a host can tell an automatic pass
        from one the user asked for while keeping one vocabulary for both
        (``compacting context…`` / ``context compacted``).

        SYNCHRONOUS, and deliberately still the path the ceiling takes. The
        pass is split into a summarize half and a commit half
        (:meth:`_summarize_for_compaction` / :meth:`_finish_compaction`) so the
        advisor-triggered path can run the expensive half off the turn, but
        this method still runs both back to back: when the context genuinely
        reaches the threshold the turn cannot safely continue, so the pass that
        relieves it must complete before the next request is built. Making the
        ceiling asynchronous would remove the one safety net that has to block.

        Cancels an in-flight background pass first (issue #413). Running
        alongside it double-bills a summarization of the same history; awaiting
        it would make this safety net wait on a call with no deadline.
        """
        await self._cancel_background_compaction()
        await self._emit(CompactionStartEvent(reason=reason))
        return await self._finish_compaction(plan, reason=reason, summarized=None)

    async def _cancel_background_compaction(self) -> None:
        """Stop an in-flight background pass so a synchronous one does not double-bill.

        Issue #413: a ceiling (or manual) pass starting while a background
        pass is summarizing used to run alongside it. The background result
        was discarded as stale, so correctness held and only spend was
        affected — two summarization calls over the same history.

        Awaiting the background pass was rejected in review: it would make
        the safety net wait on a call with no deadline. Cancelling it
        instead stops the spend we can still stop, and the synchronous pass
        proceeds immediately. Tokens already in flight are lost either way.

        Failure of the background pass is NOT a reason to skip the ceiling
        one: this method only cancels, it never decides whether the
        synchronous pass runs. A background pass that already FAILED has
        cleared the latch in its ``finally``, so this is a no-op and the
        ceiling path continues as it always did.

        Bounded wait (0.1 s), not one event-loop tick: that bound is what
        delivers ``CancelledError`` to the pass so the new summarization
        does not overlap it, then proceeds. Waiting for the provider to
        acknowledge the cancel would reintroduce the unbounded wait.
        """
        task = self._compaction_pass_task
        self._compaction_pass_task = None
        try:
            if task is not None and not task.done():
                task.cancel()
                # Bounded wait so the cancelled pass can unwind its in-flight
                # summarization before we start another. Shielded: awaiting a
                # cancelled task without a shield would CancelledError THIS
                # (ceiling) pass, which is the one that must complete. Timed:
                # if cancellation were swallowed, an unbounded await would be
                # the wait this method exists to avoid.
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    # Awaiting a cancelled task raises CancelledError even
                    # through ``shield``. That is the inner pass unwinding,
                    # which is success — unless WE were cancelled (dispose
                    # mid-ceiling), in which case the ceiling pass must not
                    # swallow it. ``Task.cancelling()`` is 3.11+; this
                    # project requires 3.12.
                    me = asyncio.current_task()
                    if me is not None and me.cancelling():
                        raise
                except Exception:
                    # The inner pass's swallows currently keep this path
                    # dead. If they ever stop, ``wait_for`` would raise
                    # into ``_run_compaction`` BEFORE the ceiling starts —
                    # the missed-compaction trap #413 named as worse than
                    # a double-bill (review round 1, F1). Cancel still
                    # happened; the ceiling must proceed.
                    pass
        finally:
            # Whatever the background pass produced is about to be stale:
            # we are committing a different pass against the live history.
            # Drop it so a later boundary cannot apply a summary of a
            # prefix we are about to replace. Also covers the race where
            # cancel was requested after the summary returned and the
            # write of ``_pending_compaction`` ran anyway (no await
            # between those two, so CancelledError is not delivered until
            # the next yield). In ``finally`` so a surprise from
            # ``wait_for`` cannot skip the latch-clear and then skip the
            # ceiling pass (F1).
            self._pending_compaction = None
            self._compaction_pass_in_flight = False

    async def _summarize_for_compaction(
        self, plan: _CompactionPlan
    ) -> tuple[str, dict[str, Any] | None]:
        """The EXPENSIVE half of a pass: the summary, and nothing else.

        Separated so it can run detached (see :meth:`_run_compaction_async`).
        It reads ``plan.llm_history``, a snapshot taken when the plan was made,
        and touches NO session state — no events, no context rebuild, no
        transcript write — so a call still in flight when the conversation
        moves on is discardable at zero cost. Every mutation lives in the
        commit half.
        """
        return await self._produce_summary(
            plan.compaction_api, plan.llm_history[: plan.cut], plan.strategy
        )

    async def _finish_compaction(
        self,
        plan: _CompactionPlan,
        *,
        reason: str,
        summarized: tuple[str, dict[str, Any] | None] | None,
    ) -> CompactionOutcome:
        """Summarize if needed, then COMMIT — the one commit, for every path.

        ``summarized`` is the pre-computed output of
        :meth:`_summarize_for_compaction` for the asynchronous path, or ``None``
        to produce it inline (the synchronous callers). One body either way, so
        an async pass and a ceiling pass cannot drift about what a commit is;
        the caller has already emitted ``CompactionStartEvent``, and every
        failure below emits the matching end event rather than leaving a host's
        "compacting context…" row open forever.
        """
        compaction_api = plan.compaction_api
        try:
            to_summarize = plan.llm_history[: plan.cut]
            # The KEPT window is rebuilt from the UNSTRIPPED render. The plan's
            # ``llm_history`` went through the capability strip (the summary
            # request must not carry images a text-only model cannot take),
            # but committing stripped messages into ``_context.messages``
            # would bake the omission into the live session: the notice text
            # would replace the image blocks for good, and switching back to
            # a vision model would restore nothing for the kept window — the
            # exact stickiness this degrade was written to avoid (agent review
            # round 1, MAJOR 1). The transcript is untouched either way, so
            # resume and ``/export`` keep their frames; this keeps the LIVE
            # context honest too. The strip still applies to what the next
            # request SENDS (``_render_history`` re-renders on the way out).
            kept = self._render_for_compaction(keep_images=True)[plan.cut :]
            if not kept:
                kept = plan.llm_history[plan.cut :]
            summary, preserve_data = (
                summarized
                if summarized is not None
                else await self._produce_summary(compaction_api, to_summarize, plan.strategy)
            )
            # STRUCTURAL guarantee that a user turn is never paraphrased away.
            # ``to_summarize`` is the RENDERED history, where a prior marker and
            # every injected user-role delivery (wake/hub/incident/todo) already
            # look like a plain user Message; the genuine prompts are the
            # ``Message(role="user")`` entries in the LIVE context (injected
            # content is a CustomMessage there), so their ids are the filter
            # that keeps a previous summary from being carried forward verbatim.
            # The summarizer still ran over ``to_summarize`` above — the summary
            # may paraphrase the user, and that is fine BECAUSE the verbatim
            # copy rides alongside it and is what the model reads.
            genuine_user_ids = {
                message.id
                for message in self._context.messages
                if isinstance(message, Message) and message.role == "user"
            }
            # Resolved off the module by NAME so a partial test double that does
            # not expose it degrades to no preservation rather than crashing the
            # pass — the same tolerance ``_offloaded`` grants its rulers. The
            # real ``compaction.api`` always exports it, so production always
            # gets the structural guarantee.
            extract: Any = getattr(compaction_api, "extract_preserved_user_turns", None)
            preserved_user_turns: list[dict[str, str]] = []
            if callable(extract):
                extracted: Any = extract(to_summarize, genuine_user_ids)
                preserved_user_turns = list(extracted)
            first_kept_entry_id = kept[0].id
            await self._transcript.append_compaction(
                summary,
                first_kept_entry_id,
                plan.context_tokens,
                preserve_data=preserve_data,
                preserved_user_turns=preserved_user_turns,
            )
            marker = build_compaction_marker(summary, preserve_data)
            # Rebuild the verbatim user turns as real user messages, reusing
            # each turn's original id so the live context and a resumed
            # ``build_llm_history`` (which re-injects the SAME payload) stay
            # byte-for-byte identical — the equivalence other tests pin.
            # ``compaction_preserved`` marks each as already-compacted
            # carried-forward content, so a subsequent pass does not re-count
            # it as fresh history (see ``cutpoint._is_preserved_user_turn``);
            # it rides ``provider_payload``, which the wire builders never ship.
            preserved_messages = []
            for turn in preserved_user_turns:
                message = _replayed_user_message([TextContent(text=turn["text"])], turn["id"])
                message.provider_payload = {compaction_api.PRESERVED_USER_TURN_KEY: True}
                preserved_messages.append(message)
            # The context becomes the RENDERED history, so a live todo reminder
            # does not survive this — by design, and the reason the plan renders
            # without them (see :meth:`_render_for_compaction`): a reminder
            # baked in here as a plain user message is past both of the guards
            # that expire it.
            # ``[marker, *preserved_user_turns, *kept]``: the marker (rendered
            # as a user message) sitting next to a preserved user turn is a
            # legal adjacency. Two consecutive user-role messages are accepted
            # by every provider wire format we target (they are not required to
            # strictly alternate on the user side), and this exact shape was
            # already reachable before this change — a marker immediately
            # followed by a user turn in ``kept`` produced ``[marker, user,
            # ...]`` already. The preserved turns carry no tool_call pairing, so
            # none of the orphan invariants can be affected.
            self._context.messages = [marker, *preserved_messages, *kept]
            history_after = await self._offloaded(
                compaction_api, "estimate_messages_tokens", self._render_for_compaction()
            )
            # The local ruler prices every image at a flat IMAGE_TOKEN_ESTIMATE
            # (1200), but the archive frames the marker just replayed are billed
            # by the PROVIDER's formula — ~5,000 visual tokens for an Anthropic
            # 1932px frame. Correct the after-figure by the difference so the
            # receipt prices the archive the way the next bill will. Measured on
            # the live session that motivated this: the uncorrected figure read
            # 60.1k against a provider-reported 137.5k on the next request.
            #
            # Computed HERE but applied AFTER the ratio below, and the order is
            # load-bearing (agent review round 1, major-2). This addend is a
            # PROVIDER-scale quantity; ``history_after`` and ``history_before``
            # are both local estimates, and the receipt's proportionality
            # argument holds only while numerator and denominator stay on that
            # one ruler. Folding it into ``history_after`` before dividing
            # would inflate the ratio and then multiply the addend by the
            # provider total a second time — on a 6-frame archive against a
            # 400k→60k pass, an after-figure of 144,900 where 105,000 plus the
            # correction is right. Adding it afterwards prices the frames once,
            # in the units the receipt is already reporting in.
            frame_correction = 0
            snap_payload = (preserve_data or {}).get("snapcompact")
            if isinstance(snap_payload, dict):
                try:
                    from local_operator.compaction.snapcompact import (
                        frame_token_estimate_for,
                    )

                    frame_count = len(snap_payload.get("frames") or [])
                    per_frame = frame_token_estimate_for(self._model.provider, self._model.model_id)
                    frame_correction = frame_count * (per_frame - IMAGE_TOKEN_ESTIMATE)
                except Exception:  # noqa: BLE001 - a receipt must not fail the pass
                    logger.debug("frame pricing correction failed", exc_info=True)
            # ``tokens_before`` is the figure the GATE acted on —
            # ``max(provider-reported, local estimate)`` — not the bare local
            # estimate. The user compares the receipt against the status band,
            # and the band shows the provider's number: a pass that fired at a
            # provider-reported 600k and then printed "319.4k → …" (the local
            # estimate) read as the band and the receipt disagreeing about
            # what just happened. omp quotes the provider figure for the same
            # reason (``calculateContextTokens(lastUsage)``).
            #
            # The after-figure must stay on the SAME ruler as that before —
            # and it cannot be reached by subtraction. The provider figure is
            # the full request (system blocks + tool schemas + history) priced
            # by the PROVIDER's tokenizer, while ``history_before``/
            # ``history_after`` price history alone with the local cl100k
            # estimator. Subtracting a local saving from a provider total
            # (the previous ``context_tokens - (tokens_before -
            # history_after)``) silently assumes those two rulers agree
            # 1-for-1. They do not. Fitting ``provider = a * local + b`` over a
            # real 10-pass session (``docs/evidence/compaction-ruler/
            # slope_fit.py``) puts ``claude-opus-5`` at slope 1.685 and
            # ``claude-opus-4-8`` at 1.622, against 1.036 for an OpenAI control
            # and 1.019 for GLM in the SAME session with the same tool schemas.
            # The per-request ratio on opus-5 runs 1.75-1.90. So the divergence
            # is the provider's tokenizer on code/JSON-dense content, not
            # content we failed to count.
            #
            # It is a per-MODEL property, which is the reason this code carries
            # no slope constant. The same fit run per EPOCH shows the three
            # single-model stretches fitting tightly (mean error 419 / 653 /
            # 1,415 tokens) while every model-switching stretch does not
            # (9,913 to 71,026) — one line cannot describe two tokenizers.
            #
            # Every locally-measured token the old form subtracted was
            # therefore worth ~1.7 provider tokens on Anthropic, and the
            # receipt understated each pass by ~140k: the screenshot that
            # motivated this read "546.5k → 419.0k (23% smaller)" for a pass
            # whose true after-figure was 311.2k, a 43% reduction.
            #
            # Scaling PROPORTIONALLY fixes that without either ruler's
            # constants having to be known. History shrank to ``history_after /
            # history_before`` of itself on the local ruler; applying that same
            # fraction to the provider's own total transports the provider's
            # fixed overhead AND its tokenizer skew across the pass, because
            # both ride along inside ``context_tokens``. It is self-calibrating:
            # no per-provider slope is stored, so a model whose tokenizer
            # differs tomorrow needs no change here. Measured over all 10 passes
            # of that session, mean absolute error against the provider's next
            # reported context falls from 139,406 tokens to 9,682 — measured
            # against the WHOLE of the arithmetic below (ratio, shrink guard,
            # frame correction after the division, and the clamp), not the
            # ratio alone.
            #
            # The approximation it makes, stated plainly: ``context_tokens``
            # is ``overhead + history``, and scaling the whole thing shrinks
            # the overhead too, though a pass never removes a system block or
            # a tool schema. That biases the estimate LOW by
            # ``overhead * (1 - history_after / history_before)``. It is
            # dominated by the tokenizer skew it corrects, and the measured
            # residual is in fact biased slightly HIGH (+4.3k to +16.2k on
            # eight of the ten passes), so subtracting a separately-estimated
            # overhead here would make the answer worse, not better. The
            # overhead is also not separable: it is the intercept ``b`` of the
            # fit above, and the per-epoch table shows ``b`` swinging from
            # -119,224 to +122,208 across model-mixed stretches — not a
            # quantity this code could pin down and subtract.
            #
            # A tuned-slope variant (``context_tokens - slope * (before -
            # after)``) was measured and is strictly worse even at its best
            # value — 19,512 at slope 1.75 — while baking in a constant that
            # rots the moment a provider retokenizes or the model mix changes.
            # Do not reintroduce it.
            #
            # The zero-guard makes the division total: a pass over empty
            # history cannot divide.
            #
            # The ``history_after`` floor is a BACKSTOP, and it is worth being
            # precise about how little it does, because a guard that reads as
            # load-bearing when it is not is how the next reader misjudges this
            # arithmetic. ``plan.context_tokens`` is
            # ``compaction_context_tokens(provider, local) = max(provider,
            # local)``, so it is always >= ``history_before``; with the
            # shrink guard below ensuring ``history_after < history_before``,
            # the product is always >= ``history_after`` and the floor cannot
            # bind through any path a real session takes. It is kept because it
            # costs nothing and because a receipt reading "0 tokens" is a worse
            # lie than a stale one if either of those invariants is ever
            # weakened — the sibling incident on the status band is documented
            # at ``tui/app.py`` ``_settle_context_reading``.
            #
            # The ratio is applied ONLY when the pass actually shrank local
            # history. That guard is not defensive tidiness: on snapcompact
            # ``history_after`` routinely EXCEEDS ``history_before`` (agent
            # review round 2, blocker-1). The pass replaces history with
            # verbatim text edges plus archive text plus images that the local
            # ruler prices at a flat ``IMAGE_TOKEN_ESTIMATE``, so the saving is
            # real on the PROVIDER ruler — that is the entire point of
            # replaying frames — while the local estimate of the replacement
            # can be larger than the original. A ratio above 1 then multiplies
            # an already ~1.7x provider total and the receipt reports a
            # compaction that GREW the context: measured at 70,888 → 111,594
            # on a real over-threshold snapcompact pass.
            #
            # There is no ratio worth applying in that state, so this does not
            # try to invent one. ``context_tokens`` is the honest answer: the
            # pass did not reduce the history this ruler can see, so the
            # receipt reports the size it already knew about rather than a
            # number derived from a ratio the measurement contradicts.
            #
            # Near-parity with the shipped subtraction form, which computed
            # ``max(history_after, context_tokens - 0)`` here. The two agree
            # whenever a provider figure has been recorded; they diverge when
            # ``history_after > context_tokens``, reachable only with no usage
            # record yet (``context_tokens`` then falls back to the local
            # estimate), where this reports ~17-18k LOWER. That is the safe
            # direction — the alternative paints above the model's window — and
            # it reaches nothing that branches on the value: in this branch
            # ``tokens_after == context_tokens`` exactly, and the recovery band
            # requires ``<= 0.8 * threshold`` while the gate only fires above
            # the threshold, so the band fails identically under both forms
            # (agent review round 3, minor-1).
            #
            # WHEN THIS BRANCH IS REACHED, measured rather than assumed:
            # ``docs/evidence/compaction-ruler/fallback_reach.py`` replays the
            # ten real snapcompact passes of the production session behind this
            # fix and **0 of 10 take it** — every one shrinks local history
            # 1.8x to 8.8x and gets the full proportional receipt.
            #
            # The mechanism is the archive's FIXED overhead: a snapcompact
            # archive carries plain-text edges sized by frame shape
            # (``HQ_EDGE_FRAMES``), not by how much history was removed —
            # roughly 20,900 tokens at the shipped Anthropic shape. Measured
            # over real passes, ``history_after - history_before`` sits at a
            # flat +21.5k from 24k to 420k of history (the edge budget, near
            # enough exactly) and only goes negative once the summarized middle
            # exceeds it.
            #
            # So the branch turns on "did this pass remove more than the edge
            # cost", which depends on how much of the history is SUMMARIZABLE
            # rather than on its size alone — which is why no single token
            # threshold is quoted here, and why the crossover lands in
            # different places in different fixtures. An earlier revision of
            # this comment generalised from synthetic 10-70 turn fixtures and
            # claimed a vision model always lands here; that was a property of
            # the fixtures, not of the model.
            #
            # Where it IS reached the bare "context compacted" line is the
            # honest report available from a local-only measurement: the
            # pass's saving is in images the provider prices several times
            # higher than this ruler does, so no locally-computed ratio can
            # see it. Making that case informative needs a provider-side
            # after-figure (the next request's usage), which arrives after this
            # event is emitted — a larger change than this one, and NOT
            # reachable by choosing different arithmetic here.
            #
            # The final clamp is the invariant the old subtraction form got for
            # free and this form does not. ``context_tokens - max(0, saved)``
            # could never exceed ``context_tokens``; a product can. Without the
            # clamp the receipt drops its own numbers (``tui/app.py``'s
            # ``compaction_receipt`` prints a bare "context compacted" when
            # ``after >= before``) and the status band paints a figure above
            # the model's whole context window.
            history_before = plan.tokens_before
            if history_before > 0 and history_after < history_before:
                scaled = round(plan.context_tokens * history_after / history_before)
                tokens_after = max(history_after, scaled)
            else:
                tokens_after = plan.context_tokens
            # The snapcompact frame correction, priced once and in the
            # receipt's own units (see where it is computed above), then the
            # whole figure bounded by what it is reporting a reduction FROM.
            tokens_after = min(plan.context_tokens, tokens_after + frame_correction)
            await self._emit(
                CompactionEndEvent(
                    reason=reason,
                    success=True,
                    strategy=plan.strategy,
                    tokens_before=plan.context_tokens,
                    tokens_after=tokens_after,
                    # A pass that fired BELOW the configured threshold owes the
                    # user an explanation, or the receipt looks like the
                    # trigger misfired. The hint is read here only to render
                    # that sentence; nothing about the pass depends on it.
                    detail=_advisor_detail(plan.advisor_hint),
                )
            )
            return CompactionOutcome(
                ran=True,
                strategy=plan.strategy,
                tokens_before=plan.context_tokens,
                tokens_after=tokens_after,
            )
        except Exception as exc:
            logger.warning("compaction failed", exc_info=True)
            await self._emit(CompactionEndEvent(reason=reason, success=False))
            return CompactionOutcome(ran=False, reason="failed", detail=f"compaction failed: {exc}")

    def _fires_on_size_alone(self, compaction_api: Any, context_tokens: int, settings: Any) -> bool:
        """Would this context have compacted with NO advice at all?

        The one place that question is asked, because two different decisions
        depend on it and they must never disagree:

        - whether a pass may be deferred (:meth:`_pass_may_run_off_the_turn`);
        - whether the advisor may be CREDITED with a pass, and judged by it
          (``advisor_hint`` on the plan).

        Asked through the same ``should_compact`` the gate itself uses, with
        ``advisory_ok=False``, so this is a reading of the one resolved trigger
        rather than a second notion of the ceiling.
        """
        return _should_compact(
            compaction_api,
            context_tokens,
            self.effective_model.context_window,
            settings,
            False,
        )

    def _pass_may_run_off_the_turn(self, plan: _CompactionPlan) -> bool:
        """Whether THIS pass may be deferred, or must relieve the turn now.

        The async path exists for a pass that fires EARLY, on advice, while the
        context is still comfortably below the configured trigger: nothing is
        forcing relief, so nobody should wait for a summarization call. The
        moment the context is genuinely at or above the ordinary ceiling that
        reasoning inverts — the turn cannot safely continue, and the pass is
        the only thing standing between it and an overflow.

        The routing therefore asks the ACTUAL CONDITION rather than "is a hint
        in hand". Those are not the same question, and agent review round 4
        (MAJOR-1) is what proved it: ``_maybe_spawn_advisor`` gates only on a
        LOWER bound (``advisor_trigger_tokens``), and ``_advisory_is_usable``
        checks ``compact_now`` and freshness only, so nothing stops a usable
        hint from being in hand while the context sits over the ceiling.
        ``should_compact`` then returns True on size ALONE, the plan carries an
        ``advisor_hint``, and a hint-presence test routed a genuine breach into
        the background: reproduced at 700k against a 600k trigger with zero
        synchronous passes, and sustained across five consecutive ceiling
        boundaries while the provider was slow. The comments here used to
        assert this could not happen, which is worse than the bug: a reader
        trusts an invariant the code never enforced.

        Expressed through :meth:`_fires_on_size_alone`, which is also what
        decides whether the advisor may be credited with a pass — the two
        answers have to come from one question, or a pass could be deferred as
        "advisory" while being attributed to size, or the reverse.
        """
        return not self._fires_on_size_alone(
            plan.compaction_api, plan.context_tokens, plan.settings
        )

    def _spawn_compaction_pass(self, plan: _CompactionPlan, *, reason: str) -> None:
        """Run an ADVISOR-triggered pass off the turn, awaiting nothing.

        CALLER CONTRACT, checked below rather than asserted in prose: the pass
        must already have been established as deferrable — advisory AND below
        the ordinary trigger (:meth:`_pass_may_run_off_the_turn`). An
        over-ceiling context must never arrive here; it takes the synchronous
        pass, which is the safety net.

        The re-check is deliberate belt-and-braces. This whole PR exists
        because a comment asserted an invariant nothing enforced (round 4,
        MAJOR-1), so the precondition that replaced it is enforced at the point
        that depends on it rather than trusted from two call sites. It is a
        cheap threshold read, and a future third caller cannot quietly
        reintroduce the bug.

        Given that, declining is a full answer and never a fall-back to the
        inline pass: below the trigger nothing is forcing relief, so skipping
        costs a later trigger while running one inline costs the user a
        summarization call mid-conversation — the stall this whole path
        removes. The callers therefore return either way.

        Only ONE detached pass may be outstanding. The gate that calls this
        fires at every tool-loop boundary, so without the latch a long tool run
        would spawn a summarization call per batch — each against a snapshot
        the next one invalidates, and all of them billed.

        The latch still only bounds BACKGROUND passes against each other. A
        ceiling pass that starts while one is in flight CANCELS it rather than
        awaiting it or running alongside it — see
        :meth:`_cancel_background_compaction`. Awaiting was independently
        rejected in agent review round 5 (#413): it would make the one
        non-deferrable pass wait on a call this design deliberately leaves
        unbounded, which is round 4's MAJOR-1 in a subtler form. Running
        alongside bills two summarizations of the same history, and the
        background result is then discarded as stale.
        """
        if self._disposed or self._compaction_pass_in_flight:
            return
        if not self._pass_may_run_off_the_turn(plan):
            # The caller contract, enforced. Deferring an at-or-above-ceiling
            # pass is MAJOR-1; refusing here means the worst a mistaken caller
            # can do is fail to spawn, and every caller already falls through
            # to the synchronous pass when this declines.
            logger.debug(
                "refusing to defer a compaction pass: the context is at or above "
                "the resolved trigger, so the pass must relieve the turn now"
            )
            return
        if self._pending_compaction is not None:
            # A finished pass is already waiting to be applied. Starting a
            # second one would summarize a prefix the pending one is about to
            # replace, guaranteeing the loser is discarded as stale.
            return
        self._compaction_pass_in_flight = True
        task = self._spawn_background(self._run_compaction_async(plan, reason=reason))
        # Dispose can race the spawn and return None; the latch still has to
        # drop or a later session would think a pass was outstanding.
        if task is None:
            self._compaction_pass_in_flight = False
            return
        self._compaction_pass_task = task

    async def _run_compaction_async(self, plan: _CompactionPlan, *, reason: str) -> None:
        """The detached half: summarize against the snapshot, park the result.

        Deliberately emits NO events and mutates no context. The pass is
        speculative until it is applied, and a host that painted
        ``compacting context…`` here would show a spinner for a pass that may
        legitimately be discarded, for however long the provider takes. The
        start/end pair is emitted around the COMMIT (see
        :meth:`_apply_pending_compaction`), where it stays adjacent and brief.

        Total, like the advisor call it descends from: every failure is a
        missing pending pass, which is exactly the state the session was in
        before. A ceiling pass that starts while this is in flight cancels
        it (issue #413) rather than running alongside; the identity check
        below is what stops a cancelled pass from parking a stale result.
        """
        try:
            summary, preserve_data = await self._summarize_for_compaction(plan)
            # Identity, not a disposed flag: a ceiling pass that cancelled us
            # has already dropped the handle (issue #413), and there is no
            # await between this return and the write below, so CancelledError
            # may not have been delivered yet. Writing pending then would park
            # a summary the ceiling pass is about to make stale. Same shape as
            # the steer-receipt guard (PR #452): drop the delivery when the
            # owner is no longer the current one.
            if self._disposed or self._compaction_pass_task is not asyncio.current_task():
                return
            self._pending_compaction = _PendingCompaction(
                plan=plan,
                summary=summary,
                preserve_data=preserve_data,
                reason=reason,
                summarized_ids=tuple(str(message.id) for message in plan.llm_history[: plan.cut]),
            )
        except asyncio.CancelledError:
            # Caught for the reason ``_run_advisor`` catches it: this task is
            # detached and routinely cancelled at dispose AND when a ceiling
            # pass starts (issue #413). A traceback for a pass nobody awaited
            # is noise. Re-raise so the wrapping ``_spawn_background`` task
            # still settles as cancelled — ``Task.cancel()`` is how the
            # ceiling path stops the call, and swallowing here would leave
            # that Task running until the provider returned.
            raise
        except Exception:  # noqa: BLE001 — a speculative pass must never break a turn
            logger.debug("asynchronous compaction pass failed", exc_info=True)
        finally:
            # Only clear the latch if it still names US. A ceiling pass that
            # cancelled this one has already dropped both, and a later spawn
            # must not have its latch stolen by our finally — the same
            # identity check as the pending write above.
            if self._compaction_pass_task is asyncio.current_task():
                self._compaction_pass_task = None
                self._compaction_pass_in_flight = False

    def _pending_is_applicable(self, pending: _PendingCompaction, live: list[Message]) -> bool:
        """Whether a finished pass still describes the conversation it planned against.

        The summary replaces ``live[: cut]``, so the pass is applicable exactly
        when that span is STILL the span it summarized — same ids, same order.
        Everything the conversation added while the call was in flight sits at
        ``live[cut:]``, which the commit keeps verbatim, so growth is safe by
        construction and needs no check of its own.

        What this rejects is the prefix having MOVED: another pass committed, a
        rebuild reordered the history, a resume replaced it. Applying then would
        splice a summary of one conversation over a different one and drop
        whatever the prefix had become — the one failure mode that must never
        be reachable, so the check is an exact identity match and the answer to
        anything else is to discard.

        Ids only, deliberately: pruning blanks tool-output CONTENT in place
        while keeping ids, and a content-sensitive check would reject a pass
        over a purely beneficial shrink.
        """
        if len(live) <= pending.plan.cut:
            # Nothing would be kept: the history is shorter than the cut the
            # plan chose, so the prefix cannot still be intact.
            return False
        return (
            tuple(str(message.id) for message in live[: pending.plan.cut]) == pending.summarized_ids
        )

    async def _apply_pending_compaction(
        self,
    ) -> tuple[_CompactionPlan, CompactionOutcome] | None:
        """Commit a finished background pass at a SAFE boundary.

        Returns the plan and outcome when a pass landed, so the post-turn gate
        can run the same recovery-band bookkeeping it runs for a synchronous
        pass; ``None`` when nothing was pending, the pass was stale, or the
        commit refused.

        Called from the two gates that already own a safe boundary: the
        mid-turn hook (a tool batch has landed, and ``_wire_legal_snapshot``
        covers the dangling-call case) and the post-turn gate. Both hold the
        turn, which is what makes rebuilding ``_context.messages`` legal here
        and nowhere else.

        The slot is cleared BEFORE the applicability check, so a stale pass is
        consumed and dropped rather than re-examined at every subsequent
        boundary — the single-use discipline ``_take_advisor_hint`` follows,
        for the same reason.
        """
        pending = self._pending_compaction
        self._pending_compaction = None
        if pending is None or self._disposed:
            return None
        live = self._render_for_compaction()
        if not self._pending_is_applicable(pending, live):
            logger.debug("discarded a background compaction pass: the conversation moved past it")
            return None
        await self._emit(CompactionStartEvent(reason=pending.reason))
        outcome = await self._finish_compaction(
            pending.plan,
            reason=pending.reason,
            summarized=(pending.summary, pending.preserve_data),
        )
        if not outcome.ran:
            return None
        self._settle_advisor(pending.plan, outcome)
        return pending.plan, outcome

    def _resolve_strategy(self, settings: Any) -> str:
        """'auto' | 'context-full' | 'snapcompact' | 'off'.

        Uses ``compaction.thresholds.resolve_strategy`` when it has landed;
        until then, an explicit ``strategy`` setting wins and ``auto`` defers
        to ``strategy_for_model``."""
        explicit = getattr(settings, "strategy", "auto")
        if explicit in ("context-full", "snapcompact", "off"):
            return str(explicit)
        try:
            from local_operator.compaction import thresholds as _thresholds

            resolver = getattr(_thresholds, "resolve_strategy", None)
            if resolver is not None:
                return str(resolver(settings, self._model))
        except ImportError:
            pass
        try:
            from local_operator.compaction.snapcompact import strategy_for_model

            return strategy_for_model(self._model)
        except ImportError:
            return "context-full"

    async def _produce_summary(
        self, compaction_api: Any, to_summarize: list[Message], strategy: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Summary text + optional ``preserve_data`` for one compaction pass.

        Snapcompact stores ``{"snapcompact": <archive dump>}`` (JSON-safe:
        base64 frames) and renders the frames back into context through the
        converter. The TEXT slot must therefore be a real short summary —
        never ``archive.text``: the archive is the full bounded history, and
        replaying it as text while the frames are dropped means the pass
        reduces nothing and re-fires on the next turn. Any error — including
        ImportError — falls back to the one-shot LLM summary.

        The snapcompact branch makes NO provider call — that is its contract
        (omp: "Archive history onto dense bitmap images the model reads back
        (no LLM call)"), and for a while this method broke it: the archive was
        built locally and then the whole discarded history was ALSO shipped to
        the provider for a written digest of frames that already carry the
        real thing. On a 600k-token session that call was 20–50 s of the
        ~60 s a manual ``/compact`` took, with the archive render spending the
        rest — against omp's near-instant pass. The text slot is now the
        deterministic reading-instructions digest (``archive_summary``), whose
        every fact derives from the archive itself.

        ``compact_to_archive`` runs in a worker thread: it is pure CPU
        (serialize → tokenize → rasterize → deflate) on the order of half a
        second, and the one event loop it would otherwise stall is shared
        with every subagent and the TUI repaint — the same reasoning as
        :meth:`_offloaded`, at a call site that cannot use it (this is not a
        ruler on the module).
        """
        if strategy == "snapcompact":
            try:
                from local_operator.compaction import snapcompact

                # The EFFECTIVE model throughout: the archive is being sized
                # and tokenized for the model that will actually receive the
                # replay, which during a fallback is the fallback.
                effective = self.effective_model
                archive = await asyncio.to_thread(
                    snapcompact.compact_to_archive,
                    to_summarize,
                    effective.provider,
                    effective.model_id,
                    self._previous_archive_text(),
                    context_window=effective.context_window,
                )
                summary = snapcompact.archive_summary(
                    archive, effective.provider, effective.model_id
                )
                return summary or " ", {"snapcompact": _archive_to_json(archive)}
            except Exception:
                logger.warning("snapcompact failed; falling back to context-full", exc_info=True)
        summary = await compaction_api.summarize_messages(to_summarize, self._one_shot_complete)
        return summary, None

    def _previous_archive_text(self) -> str | None:
        """The latest compaction's archive text, so snapcompact re-renders from
        accumulated history instead of carrying old PNGs forward."""
        for entry in reversed(self._transcript.entries()):
            if entry.type != "compaction":
                continue
            preserve = entry.payload.get("preserve_data") or {}
            snap = preserve.get("snapcompact")
            if isinstance(snap, dict) and snap.get("text"):
                return str(snap["text"])
            return entry.payload.get("summary")
        return None

    #: Output cap for :meth:`complete_once`, and the only bound on what a model
    #: that ignores the output format can bill us for. The cap counts EVERY
    #: response token, including thinking. Seven Claude naming calls at lowest
    #: effort emitted 18–38 tokens, which is why 128 used to look generous —
    #: Claude has an effort ladder, so ``_lowest_effort`` keeps thinking short
    #: and the title still fits. Families with no ladder (xAI, DeepSeek, Kimi;
    #: see ``model.effort``) can spend the whole budget thinking and emit no
    #: visible title at all, which is how Grok sessions kept the opener excerpt
    #: while still billing the thinking. omp hit the same wall (oh-my-pi #4355)
    #: and raised the title ceiling to 1024. Headroom is free: only produced
    #: tokens are billed. Do not invent an unverified ``disable_reasoning``
    #: wire key to shrink this — a 400 on the errand is swallowed and the
    #: session stays unnamed.
    ERRAND_MAX_TOKENS = 1024

    async def complete_once(self, system: str, prompt: str) -> str:
        """One CHEAP, ISOLATED, single-attempt provider call for a host errand.

        Hosts need the session's configured provider and credentials for small
        side errands — conversation auto-naming is the only caller — and
        rebuilding a client from the spec would duplicate the whole auth
        cascade. The call carries no tools, no history and no abort signal: it
        is not a turn and must not appear in the transcript.

        It used to be deferred until the turn settled, because a second
        simultaneous request at minute zero could rate-limit both. It now runs
        CONCURRENTLY with the turn, so the safety comes from the shape of the
        request instead of from the timing:

        * ``isolated`` — one attempt, no fallback chain, no credential
          rotation, no sticky-route read or write, no quota preflight, no
          effort-boundary classification, a read-only credential resolve and
          not the session's prompt cache key. See the field's docstring for the
          six pieces of session-wide state that protects, and why each one
          mattered.
        * ``replayable=False`` — deliberately the opposite of the compaction
          errand below. Replay exists so a stalled read does not permanently
          lose an EXPENSIVE result; a title is worth one attempt and no more.
        * ``max_tokens`` — bounds a model that ignores the output format.
        * cheapest route available: the ``lo`` subagent tier when the operator
          has configured one, otherwise this session's model — either way
          clamped to the lowest reasoning effort the spec accepts, because that
          token cap counts thinking tokens as well as the title.
        """
        model = self._errand_model()
        request = ChatRequest(
            model=model,
            system_blocks=[system],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
            max_tokens=self.ERRAND_MAX_TOKENS,
            # Titling is extraction, not generation, and a backend that
            # defaults temperature high otherwise garbles names — but only
            # where the model's own family accepts a value from us at all.
            #
            # Deferred to the spec's policy rather than hardcoded: a request
            # value OUTRANKS the spec in ``_sampling_params``, so a literal 0
            # here re-asserted the very number the sampling policy exists to
            # stop sending, on exactly the families it was written for (Gemini
            # 3.x, GPT-6, DeepSeek V4 all received 0.0 on this errand while the
            # main turn correctly sent nothing). ``supports_sampling_params``
            # covers the outright REJECTIONS, but not the families we merely
            # stopped asserting over, so gating on the seed is what keeps the
            # errand consistent with the turn.
            #
            # The failure mode this protects is quiet: the errand is
            # ``isolated`` with one attempt and no fallback, so a rejected
            # request surfaces only as a conversation that never gets a title.
            temperature=model.temperature,
            replayable=False,
            isolated=True,
        )
        parts: list[str] = []
        async for event in self._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
        return "".join(parts)

    def _errand_model(self) -> ModelSpec:
        """The cheapest spec this session can reach for a decorative errand,
        always on the bottom rung of whatever reasoning ladder it has.

        Prefers the operator's ``lo`` tier (``values.subagents.models.lo``),
        which is the same ladder a scout subagent runs on — an operator who has
        already said "this is my cheap model" should not have to say it twice.
        With no tier configured it uses the model ACTUALLY serving requests
        (the pinned fallback while one is in force, else the selected model).
        A naming call that stayed on the selected primary after a quota
        fallback would 429 on the dead route and leave the session untitled
        while the turn already answered on the fallback.

        The clamp is applied to WHICHEVER of the two this returns, and it is not
        only a cost argument. ``ERRAND_MAX_TOKENS`` becomes
        ``max_output_tokens``, which counts reasoning tokens too, so a spec left
        on its provider's default effort can spend the whole token budget
        thinking, emit no visible title at all and make ``parse_title`` return
        ``None`` — auto-naming would silently never produce a title for that
        operator while still billing the thinking. ``build_model_spec`` seeds
        ``reasoning_effort`` from ``default_effort``, which is ``None`` for
        several reasoning families (no ``reasoning.effort`` goes on the wire, so
        the provider applies its own default), and that is exactly how a
        configured ``lo`` tier used to reach the wire unclamped. Naming is a
        formatting job, not a thinking job. A model with no effort knob is
        unaffected.
        """
        tier = self._resolve_subagent_model("task", "lo")
        return self._lowest_effort(tier if tier is not None else self.effective_model)

    @staticmethod
    def _lowest_effort(spec: ModelSpec) -> ModelSpec:
        """``spec`` on the bottom rung of its own effort ladder, if it has one."""
        efforts = spec.reasoning_efforts
        if not efforts or spec.reasoning_effort == efforts[0]:
            return spec
        return spec.model_copy(update={"reasoning_effort": efforts[0]})

    async def _one_shot_complete(self, system: str, prompt: str) -> str:
        """One non-tool provider call used to produce the compaction summary.

        ``replayable``: nothing here reaches a screen until the whole string is
        assembled, so a transient failure part-way through the stream can be
        discarded and retried. Without it a single stalled read failed the
        compaction outright and the context it was meant to shrink kept growing.
        """
        request = ChatRequest(
            model=self._model,
            system_blocks=[system],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
            replayable=True,
            # ``0`` is a DELIBERATE hint, not a missing one: this prompt is a
            # fresh write-once system+transcript prefix, not the turn's cached
            # prefix, so a 1h entry (2x write rate) buys nothing — it is never
            # replayed except a stall retry. Inheriting the session's large
            # pre-compaction count here would send a transcript-sized prompt
            # out at the 1h rate for several dollars of pure overpayment per
            # compaction. The stream fn passes the request's hint through
            # untouched, and the Anthropic client treats 0 as below any
            # threshold.
            context_tokens_hint=0,
        )
        parts: list[str] = []
        async for event in self._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
        return "".join(parts)

    async def complete_aside(
        self,
        turns: Sequence[AgentMessage],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> str:
        """Answer a side question against the live context WITHOUT joining it.

        This is ``complete_once``'s opposite number. ``complete_once`` is for
        errands that need the provider but not the conversation (auto-naming);
        this is for questions that need the CONVERSATION — "why did you pick
        that", "are the subagents stuck" — and must not become part of it.

        So it READS everything a real turn reads (the live system blocks, so
        the goal's volatile tail matches what the agent is actually running
        under, and the live message list) and WRITES nothing: no transcript
        entry, no append to ``_context.messages``, no event fan-out. The
        caller's ``turns`` are appended for this request only. That is what
        makes the TUI's ``/btw`` overlay an aside rather than a hidden prompt —
        pressing Esc has to leave the conversation exactly as it was found.

        Nothing to do with the loop's ``Aside`` message channel a few hundred
        lines up (``queue_aside``/``_drain_asides``), which injects messages
        INTO a running turn. This is the opposite: one request that reads the
        turn and adds nothing to it.

        It does spend real tokens, and nothing here records that. The request
        carries the whole conversation, so an aside is not free; the host is
        what owns cost accounting, so ``on_usage`` hands the provider's own
        figures back rather than this method guessing at them.

        The live tool schema, with ``tool_choice="none"``: an aside that could
        edit a file would be a turn wearing a popup, and the answer is meant to
        come from context the agent already has, so nothing may be called. The
        tools are still SENT because the tools block is the FRONT of the
        provider cache prefix — tools -> system -> messages on Anthropic, and
        part of the cached request body on the OpenAI-compatible and Gemini
        wires. A real working turn sends ``list(context.tools)``
        (``harness/loop.py``), so an aside that dropped them to ``[]`` would
        change the prefix at position 0 and force the provider to re-process the
        whole conversation at full/cache-write price instead of a cache READ.
        Sending the SAME tool schema the turn sends is what keeps the aside
        warm against the turn's cached prefix. ``tool_choice="none"`` states
        the "reads the turn, calls nothing" intent, and the OpenAI-compatible
        and Gemini builders put it on the wire literally (Gemini is taught to
        honour it, since it otherwise ignores ``tool_choice`` and non-empty
        tools would newly ALLOW a call).

        On Anthropic the wire says ``auto``, NOT ``none``. The prompt-caching
        docs list ``tool_choice`` as invalidating the MESSAGES level of that
        provider's cache hierarchy (tools -> system -> messages), which made a
        differing value the prime suspect for the fleet's head-only cache
        events; measured live, ``none`` reads the turn's full prefix just as
        well (``scripts/measure_aside_tool_choice_cache.py``), so the mapping
        is hygiene against that documented rule rather than a measured saving.
        The contract is enforced HERE: the loop below consumes text and
        usage only, never a ``StreamToolCallDelta``, so a ``tool_use`` block
        in the answer is inert — nothing runs, nothing joins the history. The
        appended prompts (``ASIDE_PROMPT``, ``LOOP_JUDGE_PROMPT``) also tell
        the model to answer in text, so the case is rare to begin with.

        The one way that mapping is observable is an answer that is a tool
        call and NOTHING else — the model "answered" the question by reaching
        for ``read``. That would surface as an empty answer on the card, which
        reads as a provider fault. So when the stream carried a tool call and
        no text, the request is retried once with ``tools=[]``: the tools block
        is the front of the prefix, so this retry is a full re-process at
        write price, but it is bounded to that rare case rather than paid on
        every aside, and it gives the user an answer instead of a blank. An
        answer that mixes text and a tool call is returned as its text
        without a retry; the text is what was asked for.

        ``on_usage`` fires once per provider call, so a single aside may
        deliver TWO usage figures when that retry runs — the first call's
        (bare tool call, discarded) and the retry's. Both were paid for, so
        hosts must add them rather than keep the last one.

        Safe to call mid-turn, and the pairing below is what makes that true —
        see :meth:`_wire_legal_snapshot`.
        """
        blocks = self._system_blocks()
        if inspect.isawaitable(blocks):
            blocks = await blocks
        messages = self._render_history([*self._wire_legal_snapshot(), *turns])
        request = ChatRequest(
            model=self._model,
            system_blocks=list(blocks),
            messages=messages,
            # Live tools (not []): keep the aside on the SAME cache prefix the
            # working turn builds. See the docstring for why this is a cache
            # read rather than a full re-process, and why Anthropic puts the
            # turn's own tool_choice on the wire rather than this "none".
            tools=list(self._context.tools),
            tool_choice="none",
            # Same prefix as the turn, so the same TTL: the session stamps its
            # own hint here because the shared stream fn holds none (a child
            # would overwrite it — see ``ChatRequest.context_tokens_hint``).
            context_tokens_hint=self._context_tokens_hint,
        )
        parts: list[str] = []
        called_tool = False
        async for event in self._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
                if on_delta is not None:
                    on_delta(event.delta)
            elif isinstance(event, StreamToolCallDelta):
                # Inert by design (see the docstring): recorded only so an
                # answer that was NOTHING BUT a call can be retried below.
                called_tool = True
            elif isinstance(event, StreamUsageEvent) and on_usage is not None:
                on_usage(event.usage)
        if parts or not called_tool:
            return "".join(parts)
        # Tool call and no text: the model tried to act instead of answering.
        # Retry once with no tools at all — off the cache prefix, but bounded
        # to this case. ``tools=[]`` never reaches the Anthropic mapping, so
        # the wire genuinely offers nothing to call.
        logger.debug("aside answered with a bare tool call; retrying without tools")
        retry = request.model_copy(update={"tools": []})
        async for event in self._stream_fn(retry, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
                if on_delta is not None:
                    on_delta(event.delta)
            elif isinstance(event, StreamUsageEvent) and on_usage is not None:
                on_usage(event.usage)
        return "".join(parts)

    async def advise_compaction(self, turns: Sequence[AgentMessage]) -> str:
        """One off-loop request asking the model WHEN to compact (BETA).

        Modelled directly on :meth:`complete_aside` — live system blocks, the
        wire-legal snapshot of the live history, the live tool schema with
        ``tool_choice="none"`` — and for the same reason: it is a question
        about the CONVERSATION that must not join it. Read that method's
        docstring for why the tools are sent rather than dropped; the whole
        economic case below rests on it.

        Two deliberate deltas from an aside:

        ``replayable=True`` (an aside is not). Nothing here reaches a screen
        until the whole string is assembled, so a stalled read can be
        discarded and retried rather than losing the call — the same reasoning
        as :meth:`_one_shot_complete`. Nobody is waiting, so a retry costs
        latency nobody is measuring.

        ``isolated=False``, and this one is load-bearing enough to be the
        reason the feature is worth shipping at all. ``isolated=True`` strips
        the session's ``prompt_cache_key`` on the OpenAI wire (see
        ``ChatRequest.isolated``), which puts this request on its own cache
        namespace with a 100% uncached prefix. Measured on the session that
        motivated this feature, 92.9% of prompt-side tokens were cache READS;
        an advisor call that hits the turn's warm prefix costs about 2.6% of
        the bill, and the same call on a cold namespace costs about 25.6% and
        turns the whole feature into a net loss. So the advisor deliberately
        forgoes isolation's protections (single attempt, no route/credential
        state) to stay on the session's cache key.

        For exactly the same reason there is NO ``advisor_model`` /
        ``advisor_effort`` config key, and adding one would be a regression
        rather than a feature. A cheap model looks like the obvious economy
        and is strictly worse: it has its own cache namespace, so its prefix
        is 100% uncached, and a full uncached read of a 500k context on a
        cheap model costs more than a cached read of the same context on the
        expensive one. The knob would invite users to silently destroy the
        economics that justify the call.

        MEASURED, not assumed. Three findings shape the code, and the third
        corrected the first two:

        - The system blocks are passed through UNCHANGED. The advisor's
          instructions ride inside the appended user turn instead, because
          system sits in the cache prefix ahead of the messages: adding one
          block there measured 0% cache hit and a full ``cache_write=14590``
          (``scripts/measure_advisor_cache.py``). The request must stay
          APPEND-ONLY relative to the turn's prefix.
        - On Anthropic, ``isolated=True`` measured 100% cache hit as well,
          because that provider keys caching on prefix CONTENT rather than on
          ``prompt_cache_key``. Isolation is still declined, since the key
          does govern the OpenAI-compatible wire and this method has to be
          correct on every provider a session may be running.
        - The original 96.1%-hit figure came from a ~14k-token toy
          conversation whose system+tools head WAS most of the prompt, so it
          could not distinguish a head-only hit from a full one. The shared
          context therefore blamed ``tool_choice="none"`` for the fleet's
          head-only cache events; measured live at ~37k tokens
          (``scripts/measure_aside_tool_choice_cache.py``), a ``none`` aside
          reads the turn's full prefix exactly as well as an ``auto`` one, so
          that attribution did not hold — the fleet events are better
          explained by 5-minute TTL expiry and per-sub-context prefix
          extension. The wire still sends the turn's own ``auto`` when tools
          are present, as hygiene against the documented rule that
          ``tool_choice`` invalidates the messages cache; the "calls nothing"
          contract is kept by this method reading only text and usage events.
          See
          ``docs/evidence/compaction-advisor/aside-tool-choice-measurement.txt``
          for the numbers.
        """
        blocks = self._system_blocks()
        if inspect.isawaitable(blocks):
            blocks = await blocks
        request = ChatRequest(
            model=self._model,
            system_blocks=list(blocks),
            messages=self._render_history([*self._wire_legal_snapshot(), *turns]),
            # Live tools, same as an aside: the tools block is the FRONT of the
            # provider cache prefix, so sending [] would change position 0 and
            # force a full re-process at write price instead of a cache read.
            tools=list(self._context.tools),
            tool_choice="none",
            replayable=True,
            isolated=False,
            # Append-only on the turn's prefix, so it shares the turn's TTL;
            # stamped here for the reason ``complete_aside`` gives.
            context_tokens_hint=self._context_tokens_hint,
        )
        parts: list[str] = []
        # Text and usage ONLY. A tool-call delta is dropped on the floor: on
        # Anthropic the wire says ``auto`` (see the docstring), and this is the
        # half of the contract that guarantees the advisor never acts. No
        # bare-tool-call retry here, unlike ``complete_aside``: an unparseable
        # answer is already a handled outcome (``parse_hint`` returns None and
        # the size trigger stands), and nobody is looking at a blank card.
        async for event in self._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
            elif isinstance(event, StreamUsageEvent):
                # The cache-hit rate IS the feature's cost model, so it is
                # logged rather than assumed. A run of advisor calls showing
                # low cache_read_tokens means the prefix is not being reused
                # and the advisor should be switched off.
                usage = event.usage
                logger.debug(
                    "compaction advisor usage: context=%s cache_read=%s cache_write=%s",
                    getattr(usage, "context_tokens", None),
                    getattr(usage, "cache_read_tokens", None),
                    getattr(usage, "cache_write_tokens", None),
                )
        return "".join(parts)

    def _advisor_settings(self) -> Any | None:
        """Compaction settings when the advisor may run this turn, else ``None``.

        Every gate that does not need the history lives here, so the spawn
        site stays a single readable condition and the expensive checks
        (rendering, token estimates) never run for a session that has the beta
        switched off — which is every session by default.
        """
        if self._disposed or self._advisor_disabled or self._advisor_in_flight:
            return None
        settings = self._compaction_settings
        if settings is None or not getattr(settings, "advisor_enabled", False):
            return None
        if not settings.enabled or getattr(settings, "strategy", "") == "off":
            return None
        if self._advisor_calls >= int(getattr(settings, "advisor_max_calls", 0) or 0):
            return None
        if self._generation < self._advisor_cooldown_until:
            return None
        every = int(getattr(settings, "advisor_every_n_turns", 0) or 0)
        if every > 0 and self._generation - self._advisor_last_turn < every:
            return None
        return settings

    def _maybe_spawn_advisor(self) -> None:
        """Fire an advisor call at a tool-loop boundary, awaiting nothing.

        Called from the mid-turn hook, which is the one place in a session
        that is guaranteed to be at a SAFE boundary (a tool batch has landed;
        ``_wire_legal_snapshot`` handles the dangling-call case anyway) and is
        also where the context is growing fastest.

        Nothing awaits the result. A hint that lands after the plan gate has
        already run is simply the next gate's input, and a hint that lands
        after the conversation has moved on is discarded as stale — the same
        posture ``session.naming`` takes for a late title, and the reason this
        can never add latency to a turn.
        """
        settings = self._advisor_settings()
        if settings is None:
            return
        provider_reported = (
            self._last_usage.context_tokens if self._last_usage is not None else None
        )
        trigger = int(getattr(settings, "advisor_trigger_tokens", 0) or 0)
        # Below the advisor's own trigger there is no decision to inform, and
        # the call would be pure cost. The provider figure is used rather than
        # a local estimate because this runs on the loop and a full tokenize
        # here would be exactly the stall the plan gate's upper-bound pre-gate
        # exists to avoid.
        if provider_reported is None or provider_reported < trigger:
            return
        self._advisor_in_flight = True
        self._advisor_calls += 1
        self._advisor_last_turn = self._generation
        self._spawn_background(self._run_advisor(settings, self._generation))

    async def _run_advisor(self, settings: Any, turn_index: int) -> None:
        """The advisor call, bounded and total: every failure is a no-op.

        ``_spawn_background`` already logs rather than raises, so this only has
        to guarantee the in-flight latch is released and that no provider
        failure, timeout, or malformed answer becomes anything more than a
        missing hint. The turn running alongside must never learn this
        happened.
        """
        try:
            from local_operator.compaction import api as compaction_api

            timeout = float(getattr(settings, "advisor_timeout_s", 30.0) or 30.0)
            history = list(self._context.messages)
            provider_reported = (
                self._last_usage.context_tokens if self._last_usage is not None else 0
            )
            threshold = compaction_api.resolve_threshold_tokens(
                self.effective_model.context_window, settings
            )
            prompt = compaction_api.build_advisor_prompt(
                history,
                context_tokens=provider_reported or 0,
                threshold_tokens=threshold,
            )
            turns = [Message.user(prompt)]
            # One budget for the whole call, the way _ask_for_title bounds
            # naming: there is no retry above this and the request's own
            # replay is inside it.
            raw = await asyncio.wait_for(
                self.advise_compaction(turns),
                timeout,
            )
            genuine_user_ids = {
                message.id
                for message in self._context.messages
                if isinstance(message, Message) and message.role == "user"
            }
            # ``floor_cap`` is the THIRD consumer of ``_advisor_floor_cap``,
            # and the only one where the cap moves an ACCEPT/REJECT boundary
            # rather than clamping a number (agent review round 1, major-1).
            # ``validate_hint`` rejects a hint narrower than
            # ``max(keep_recent_tokens, task_boundary_floor(..., cap))``, so a
            # lower cap lowers that floor and admits hints it used to refuse:
            # on a 1M window with a 131,376-token task span the floor drops
            # 131,376 → 100,000, and a 120,000-token hint flips from
            # "narrowing, rejected" to accepted.
            #
            # That is a widening of what the ADVISOR may propose, not of what a
            # size pass keeps, and it is bounded on both sides: a hint below
            # ``keep_recent_tokens`` is still rejected, and an accepted hint is
            # still clamped by the same cap in ``_plan_compaction``. It can
            # nonetheless make a pass fire that would not otherwise have
            # fired — ``should_compact`` takes ``min(threshold,
            # resolve_advisor_floor_tokens(...))`` once advice is usable — so
            # "nothing changes when compaction fires" is true of the SIZE
            # trigger only. The whole path is inert while ``advisor_enabled``
            # is false, which is the shipped default.
            hint = compaction_api.validate_hint(
                compaction_api.parse_hint(raw),
                history,
                genuine_user_ids=genuine_user_ids,
                min_confidence=float(getattr(settings, "advisor_min_confidence", 0.6) or 0.0),
                keep_recent_tokens=int(settings.keep_recent_tokens),
                floor_cap=self._advisor_floor_cap(settings),
                turn_index=turn_index,
            )
            if hint is not None:
                self._advisor_hint = hint
                logger.debug(
                    "compaction advisor hint: preserve=%s tokens=%d now=%s conf=%.2f",
                    hint.preserve_from_id,
                    hint.preserve_tokens,
                    hint.compact_now,
                    hint.confidence,
                )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Cancelled is caught for the reason naming catches it: this task
            # is detached and routinely cancelled at dispose, and a teardown
            # traceback for a feature nobody waited on is noise.
            pass
        except Exception:  # noqa: BLE001 — an advisory must never break a turn
            logger.debug("compaction advisor call failed", exc_info=True)
        finally:
            self._advisor_in_flight = False

    def _advisor_floor_cap(self, settings: Any) -> int:
        """Cap for ``task_boundary_floor`` — the preserve window may not eat
        the whole context.

        Above the cap a "preserved" window leaves the pass nothing to
        summarize, so ``find_cut_point`` answers ``None`` and the protection
        turns into "never compact" — the failure the trigger exists to prevent.

        Two terms, and the cap is the SMALLER of them:

        1. ``keep_recent_tokens * _TASK_FLOOR_KEEP_MULTIPLE`` — a task-shaped
           bound, on the same local ruler as the span being capped.
        2. ``threshold // 2`` — a capacity-shaped CEILING, so the cap can
           never approach the model's own context window.

        Term 1 is the fix. The cap used to be term 2 alone, which mixed two
        rulers exactly the way the receipt did: ``task_boundary_floor`` sums a
        span with the local cl100k estimator (``cutpoint.py``), while
        ``resolve_threshold_tokens`` returns a PROVIDER-scale number, and the
        two diverge by ~1.6-1.7x on Anthropic (see the slope measurement at the
        receipt in :meth:`_run_compaction`). On a 1M window that made the cap
        300,000 local tokens, and a pass whose last genuine user turn sat
        131,376 tokens back widened ``keep_recent`` 20k → 131k and retained
        41.3% of history where seven other passes of the same session retained
        4-19%.

        **Term 2 is retained deliberately, and dropping it was a bug caught in
        agent review round 1 (blocker-1).** A cap expressed only in
        ``keep_recent`` multiples is independent of the model's capacity, which
        is worse than being on the wrong ruler: at the 20,000 default a flat
        100,000 EXCEEDS the entire context window of a 32k or 64k model, and
        65% of the shipped registry's 118 ``context_window`` entries sit below
        the ~250k crossover where the flat term stops being the tighter of the
        two (median model: 131,072). Measured on the real
        ``task_boundary_floor`` → ``find_cut_point`` path, the flat-only cap
        made a 64k model refuse EVERY over-threshold pass (47/47 samples, where
        the capacity cap refused none) and dropped a 128k model's reclaim from
        52% to 7% — under ``_ADVISOR_MIN_RECLAIM_FRACTION``, which permanently
        disables the advisor for that session. That is precisely the
        "never compact" failure named above, reached through the guard meant
        to prevent it.

        Term 2 is a provider-scale number bounding a local-scale span, so it is
        a ruler mix — but as a one-sided CEILING it is a conservative one: it
        can only ever make the preserve window SMALLER, and a too-small
        preserve window costs task context while a too-large one costs the
        ability to compact at all. The proportional receipt eliminated a ruler
        mix that decided a REPORTED NUMBER; this one only bounds a safety
        margin, and no figure shown to the user or used in an equality test
        derives from it.

        ``_TASK_FLOOR_KEEP_MULTIPLE`` is 5, sized against every active-task
        span measured on real sessions: the seven recorded at
        ``cutpoint.task_boundary_floor`` plus the ten in
        ``docs/evidence/compaction-ruler/retention_real2.txt``. Pooled, those
        17 spans run p50 47.4k, p75 53.7k, p90 125.9k, max 131.4k. The
        distribution is bimodal — thirteen spans under 54k, four between 113k
        and 132k — so the choice is really "where between the two clusters
        does the bound sit".

        5x the 20,000 default puts it at 100,000: comfortably above the entire
        lower cluster (the longest ordinary task is 53,732, so the cap has
        roughly 2x headroom over it) and below the upper cluster, which is
        exactly the set of passes that retained 35-41% of history where every
        other pass retained 4-19%. Clipping those four is the bound doing its
        job, not a regression.

        **Honest limit of the evidence for this exact value.** Because the gap
        between the clusters is wide, every multiple from 3x to 5x clips the
        same four spans — no measurement here separates them, and the reason to
        prefer 5 is margin over the observed ordinary maximum rather than a
        different outcome. What the data DOES settle is the upper bound: 6x
        (120,000) lets the 113,835 span through, and 7x lets all four through,
        so the multiple must not be raised without new evidence. The full table
        is ``docs/evidence/compaction-ruler/span_percentiles.txt``.

        The outer ``max(keep_recent_tokens, ...)`` is kept, and with term 2
        back it is a real guard again rather than the tautology a
        multiple-only cap made of it: on a small window ``threshold // 2`` can
        fall BELOW ``keep_recent_tokens`` (13,107 against a 20,000 default on a
        32k model), and a cap under the verbatim window the user configured
        would silently narrow what they asked to keep.

        THREE call sites read this cap, not two:
        ``task_boundary_floor`` in :meth:`_plan_compaction`, the advisor-hint
        clamp beside it, and ``validate_hint``'s ``floor_cap``
        (``compaction/advisor.py``) — see the note in :meth:`_run_advisor` for
        the acceptance-boundary shift the third one implies. On a 1M window all
        three tighten from 300,000 to 100,000; below the ~250k crossover all
        three are unchanged from today, because term 2 binds there and term 2
        is what shipped. Both statements describe the HEALTHY path — the two
        ``except`` branches below answer 0 and ``keep_recent`` respectively,
        which is narrower than either, deliberately.
        """
        try:
            # ``max(0, ...)``: the field is not constrained positive by
            # pydantic, and a negative would otherwise be returned unchanged —
            # leaving the outer ``max`` with nothing to floor, which is the one
            # property the rest of this docstring leans on.
            keep_recent = max(0, int(settings.keep_recent_tokens))
        except Exception:  # noqa: BLE001 — degrade to plain recency
            # A partial settings double (the same tolerance ``_offloaded``
            # grants its rulers) must not break a pass. Cap 0 makes
            # ``task_boundary_floor`` return 0, the hint clamp 0 and
            # ``validate_hint``'s local floor ``keep_recent_tokens``, and every
            # caller folds that through ``max(keep_recent, ...)`` — so the
            # degraded answer is plain recency, the pre-floor behaviour.
            return 0
        task_cap = keep_recent * _TASK_FLOOR_KEEP_MULTIPLE
        try:
            from local_operator.compaction import api as compaction_api

            capacity_cap = (
                compaction_api.resolve_threshold_tokens(
                    self.effective_model.context_window, settings
                )
                // 2
            )
        except Exception:  # noqa: BLE001 — no capacity term, no widening
            # Without a capacity term there is no safe way to widen, so this
            # does not widen at all: ``keep_recent`` alone is what the
            # pre-existing implementation returned here, and it is the answer
            # that cannot fail dangerously.
            #
            # Returning the TASK term instead would restore precisely the
            # capacity-independent cap this method exists to avoid, on the one
            # path whose purpose is to fail safe — measured looser than shipped
            # on 9 of 9 windows, and reproducing ``find_cut_point`` -> ``None``
            # (never-compact) at 128k (agent review round 2, major-1). The
            # asymmetry is the point: a too-narrow preserve window costs task
            # context on a pass that still runs, while a too-wide one costs the
            # ability to compact at all.
            return keep_recent
        return max(keep_recent, min(task_cap, capacity_cap))

    def _wire_legal_snapshot(self) -> list[AgentMessage]:
        """A copy of the live message list that a provider will actually accept.

        The live list is NOT always legal. ``AgentLoop`` appends the assistant
        message the moment the model turn ends (``loop._run`` → ``context
        .messages.append(assistant)``) and appends the tool results only once
        ``_execute_tool_calls`` returns — so for the whole duration of every
        tool batch, which is the longest part of a turn, the list ends in an
        assistant message whose ``tool_calls`` have no answers. Sending that
        is a 400 on both wires ("must be followed by tool messages responding
        to each tool_call_id"; ``tool_use`` without ``tool_result``), and
        mid-batch is exactly when someone asks "what are you doing?".

        So the dangling calls are paired HERE, in a request-scoped copy, the
        same way the loop pairs them on its abort path. The placeholder text
        is also the honest answer to the question being asked.
        """
        snapshot = list(self._context.messages)
        tail = snapshot[-1] if snapshot else None
        if isinstance(tail, Message) and tail.role == "assistant" and tail.tool_calls:
            snapshot.extend(
                Message(
                    role="tool",
                    content=[TextContent(text="(still running)")],
                    tool_call_id=call.id,
                    tool_name=call.name,
                )
                for call in tail.tool_calls
            )
        return snapshot

    async def adopt_aside(self, messages: Sequence[Message]) -> None:
        """Promote an off-the-record aside exchange into the conversation.

        The one door out of :meth:`complete_aside`'s no-trace contract, and it
        is the USER's to open: an aside the user decides was worth keeping is
        appended as ordinary user/assistant turns, to the live context and the
        transcript both, so the next real turn sees it and a resume replays it.

        Refused while a turn is running, and not as caution. The loop owns
        ``_context.messages`` for the duration and pairs every tool call with
        its result; splicing a user message between an assistant tool-call
        message and the results it is waiting for produces a message list no
        provider will accept.

        The turn LOCK is consulted as well as the streaming flag, for the
        reason :meth:`prompt` spells out: ``_is_streaming`` covers only
        ``_run_turn`` itself, while the lock is held across the whole pipeline
        including a post-compaction auto-continuation. A fork landing in that
        gap would be swept into a continuation the user never saw.
        """
        if self._is_streaming or self._turn_lock.locked():
            raise RuntimeError("cannot adopt an aside while a turn is running")
        # Persist first, then adopt — the order ``seed_history`` uses, and for
        # the same reason: a failed transcript write must not leave the live
        # context carrying messages a resume will not replay.
        for message in messages:
            await self._transcript.append_message(message)
        # One synchronous extend, not an append per await. ``_deliver_wake``
        # spawns a turn precisely when nothing is streaming, so an await
        # between the pair's two messages is a window for a wake's turn to
        # splice its own messages through the middle of them.
        self._context.messages.extend(messages)

    async def record_shell(self, command: str, result: ToolResult) -> None:
        """Persist a user-typed bang-mode command into the conversation.

        The TUI runs the command itself (it is not a turn) and then asks the
        session to remember it, so the next prompt — and a resume — can see
        what just happened. omp and opencode both do this: a `! git status`
        the user ran is context, not a secret.

        A receipt that lands while a turn or manual compaction owns the message
        list is QUEUED, never discarded: splicing it into a live provider batch
        would produce an unsendable list, while dropping it would make a command
        the user can see disappear on resume. The lock owner flushes the FIFO at
        its safe boundary before another prompt can start.
        """
        if self._is_streaming or self._turn_lock.locked():
            self._pending_shell_records.append((command, result))
            return
        # Serialize the idle write too. A prompt can otherwise acquire the lock
        # while transcript.append_message is awaiting and build its request from
        # a half-written synthetic exchange. Re-check the FIFO after acquisition
        # to flush anything that raced us while we waited.
        async with self._turn_lock:
            # Queue-then-flush rather than a direct write: the FIFO pops only
            # after a successful transcript write, so a failure here leaves the
            # receipt for the next boundary instead of losing it.
            self._pending_shell_records.append((command, result))
            await self._flush_shell_records()

    async def _persist_shell_record(self, command: str, result: ToolResult) -> None:
        """Append one synthetic bash exchange to transcript and live context."""
        # Synthetic assistant+tool pair rather than a single user dump: the
        # TUI's resume replay already knows how to mount a ToolCard from a
        # call and its result, and a wall of stdout attributed to the user
        # would read as something they typed.
        user = Message.user(f"! {command}")
        assistant = Message.assistant("")
        assistant.tool_calls = [
            ToolCall(id=result.tool_call_id, name="bash", arguments={"command": command})
        ]
        tool = Message.tool_result(result)
        messages = [user, assistant, tool]
        for message in messages:
            await self._transcript.append_message(message)
        self._context.messages.extend(messages)

    async def _flush_shell_records(self) -> None:
        """Persist queued bang-mode receipts in completion order.

        Called while ``_turn_lock`` is held at the turn, wake, compaction and
        prompt boundaries, and once more from :meth:`dispose` after the app has
        settled the shell worker — the lock serializes the boundary callers,
        and teardown ordering serializes the dispose one. Pop AFTER each
        successful write so a transcript failure leaves the unpersisted tail
        available to the next boundary or dispose.
        """
        while self._pending_shell_records:
            command, result = self._pending_shell_records[0]
            await self._persist_shell_record(command, result)
            self._pending_shell_records.pop(0)

    # -- wakes -------------------------------------------------------------------

    def _load_wake_schedules(self) -> None:
        # A wake is active ownership, not conversation history. A fork declines
        # only snapshots copied at its creation boundary; snapshots appended
        # afterwards belong to the fork and must survive its next resume.
        from local_operator.fork import fork_instant

        entry = self._transcript.latest_custom_entry(WAKE_SCHEDULES_CUSTOM_TYPE)
        if entry is None:
            return
        forked_at = fork_instant(self._transcript.directory)
        if forked_at is not None and not entry.ts > forked_at:
            return
        details = dict(entry.payload.get("details", {}))
        if not details:
            return
        schedules: list[WakeSchedule] = []
        for raw in details.get("schedules", []):
            try:
                schedules.append(WakeSchedule.model_validate(raw))
            except Exception:
                logger.warning("dropping malformed persisted wake schedule: %r", raw)
        self._wake.load(schedules)

    async def _wake_deliver_via_hook(self, due: DueWake) -> None:
        """Scheduler-facing deliver trampoline: reads the CURRENT hook at fire
        time, so swapping the hook (resume catch-up shim) takes effect without
        rebuilding the scheduler."""
        await self._wake_deliver_hook(due)

    def _prepare_missed_wake_catchup(self) -> None:
        """Snapshot the overdue schedules load() just adopted and compose the
        single aggregated catch-up prompt for them. Runs in ``__init__`` so the
        state exists even when no host ever calls async_init — a session that
        only runs one prompt() still owes its catch-up."""
        missed = self._wake.take_missed()
        if not missed:
            return
        now = int(time.time() * 1000)
        # An overdue schedule has just been re-armed to now + LOAD_GRACE_MS,
        # so any overdue wake is still in grace until that deadline. Wall time
        # is good enough here: grace comparisons happen against the same clock
        # the scheduler reads (``now=lambda: int(time.time() * 1000)``).
        self._resume_grace_ends_ms = float(now + LOAD_GRACE_MS)
        # The delivery note shows the DUE-while-down count (how often the wake
        # actually came due), not the budget-clamped delivered-miss count —
        # for a limit-bounded recurring wake the clamped figure understates
        # the downtime (review round 2, M2).
        self._missed_wake_occurrences = {m["schedule"].id: m["due"] for m in missed}
        # The ids the catch-up text AGGREGATES. The shim uses this to swallow
        # only these schedules' fires — a wake that comes due later is NOT in
        # the folded text, so swallowing it would lose its message entirely
        # (review round 3, M1).
        self._resume_catchup_ids = {m["schedule"].id for m in missed}
        self._resume_catchup_text = self._format_missed_wake_catchup(missed, now)
        # Install the suppression shim NOW, not at first trigger: installing it
        # late is what let the scheduler's own tick swallow a per-schedule
        # delivery before any trigger had run (review round 2, M1).
        self._wake_deliver_hook = self._deliver_wake_catchup

    def _handle_missed_wakes(self) -> None:
        """Send the aggregated catch-up for the overdue wakes adopted at load.

        Reachable from every path that can fire a wake, because the session
        cannot know which one a host uses: ``async_init`` (the off-loop boot),
        the ``needs_rearm`` first-turn re-arm, and the head of EVERY
        ``prompt``/``_prompt_messages`` (the only trigger guaranteed to exist
        on a session built inside a running loop, where the TUI lives — the
        gap that let a resumed TUI session fire N separate wake turns and
        never aggregate, review round 2 M1). Sends once, the moment grace has
        expired, so a fresh session shows the wake LIVE rather than inside
        the boot path; the shim installed at prepare swallows the scheduler's
        per-schedule fires until then, so a resume with three overdue wakes
        costs one turn, not three."""
        catchup = self._take_resume_catchup()
        if catchup is not None:
            self._deliver_resume_catchup(catchup)

    def _take_resume_catchup(self) -> CustomMessage | None:
        """Build the catch-up message once grace has passed, else None.

        Side-effecting even when it returns None-for-now: a send that IS due
        flips ``_resume_catchup_sent`` and clears the per-schedule missed-count
        map (so a recurring wake's later punctual fires are not re-annotated
        with a stale count — review round 3, m2). Split from the delivery so a
        caller about to run a turn can take the message and INLINE it ahead of
        its own (see ``prompt``) instead of spawning a turn that would land
        behind it (review round 3, M2).
        """
        if self._resume_catchup_sent or self._resume_catchup_text is None:
            return None
        if int(time.time() * 1000) < self._resume_grace_ends_ms:
            return None
        text, self._resume_catchup_text = self._resume_catchup_text, None
        self._resume_catchup_sent = True
        self._missed_wake_occurrences = {}
        # Clear the fold set too, so the post-send shim is a passthrough by
        # STRUCTURE (empty set), not only by the _resume_catchup_sent guard.
        self._resume_catchup_ids = set()
        # The receipt event fires HERE, at take time, so both delivery modes
        # (own turn via ``_deliver_resume_catchup``, or inlined ahead of a user
        # turn in ``prompt``) paint the expandable catch-up line. It is a
        # CATCH-UP marker: the front end folds the whole set into one line, and
        # the replay loop skips re-adding the user-attributed message
        # (``wake_catchup``), so this event is the only place the missed wakes
        # are surfaced.
        self._emit_nowait(WakeDeliveredEvent(text=text, catchup=True))
        return CustomMessage(
            custom_type=WAKE_PROMPT_MESSAGE_TYPE,
            attribution="user",
            details={"wake_catchup": True, "text": text},
        )

    def _deliver_resume_catchup(self, catchup: CustomMessage) -> None:
        """Deliver the catch-up as its own turn (or a steering message mid-turn).

        Used when the catch-up fires with no user turn already in flight; the
        receipt event was already emitted by ``_take_resume_catchup``.
        """
        if self._is_streaming:
            # Same courtesy as a live wake delivery: never a reason to cancel
            # the running tool, and the text must carry the resume guidance
            # because the turn the catch-up lands in has work to return to.
            catchup.details["text"] = self._append_busy_resume_note(str(catchup.details["text"]))
            self._courtesy_wake_count += 1
            self._steering_queue.put_nowait(catchup)
            # Same wake-a-parked-`wait` mark as _deliver_wake, same ordering.
            self._peer_arrival.mark(WAKE_PROMPT_MESSAGE_TYPE)
            return
        self._spawn_background(self._prompt_messages([catchup]))

    async def _deliver_wake_catchup(self, due: DueWake) -> None:
        """Deliver hook while the resume catch-up is pending.

        Swallows ONLY the fires the catch-up text aggregates (``due.schedule.id
        in _resume_catchup_ids``): those are covered by the folded prompt, and
        the schedule is still advanced + persisted by pump, so nothing
        re-fires. Any OTHER fire — one that came due after load, so its message
        is not in the folded text — falls through to a normal delivery;
        swallowing it would lose the message entirely (review round 3, M1).
        After the catch-up is sent the hook is a plain passthrough.
        """
        if not self._resume_catchup_sent and due.schedule.id in self._resume_catchup_ids:
            return  # folded into the pending catch-up; pump already advanced it
        await self._deliver_wake(due)

    @staticmethod
    def _format_missed_wake_catchup(missed: list[MissedWakeOccurrence], now_ms: int) -> str:
        """One wake_prompt text aggregating every schedule that came due while
        the process was down. Dedup is structural, not textual: the scheduler
        collapses a recurring wake's skipped occurrences to a count (the 5
        identical "check the build" prompts from a 5-hour sleep arrive as
        ONE entry reading "5 occurrences were missed"), and each remaining
        schedule contributes its verbatim message exactly once."""

        def _due_label(schedule: WakeSchedule) -> str:
            due = datetime.fromtimestamp(schedule.next_due_at / 1000).astimezone()
            return due.strftime("%Y-%m-%d %H:%M")

        lines = []
        for entry in missed:
            schedule = entry["schedule"]
            # The DUE-while-down count is what the agent should read (M2).
            occurrences = entry["due"]
            if occurrences > 1:
                # occurrences > 1 implies a repeating schedule, so every_ms is
                # set; the assert turns that reasoning into a type narrowing.
                assert schedule.every_ms is not None
                header = (
                    f"- {schedule.id} (every {format_duration(schedule.every_ms)}, "
                    f"first missed at {_due_label(schedule)}): "
                    f"{occurrences} occurrences were missed while the session was down."
                )
            else:
                header = (
                    f"- {schedule.id} (due {_due_label(schedule)}): "
                    "missed while the session was down."
                )
            lines.append(f"{header}\n  Message: {schedule.message}")
        return (
            "(alarm) The session resumed after being closed; the following scheduled wake(s) "
            "came due while it was down. Each fires once now — handle them as missed wakes, "
            "checking the CURRENT state rather than replaying each past occurrence.\n\n"
            + "\n".join(lines)
        )

    async def _persist_wake_schedules(self, schedules: list[WakeSchedule]) -> None:
        """The ONE writer of schedule state: transcript first, then the
        derived index, then the install-on-demand hook.

        Order is the contract. The transcript ``wake_schedules`` entry is the
        source of truth and is the only step allowed to fail this coroutine:
        the scheduler's ``update()`` awaits it before re-arming, so a failed
        append means the in-memory schedules never diverge from disk. The
        index write and the install hook run *after* it and are wrapped so
        they can never turn a persisted schedule into a raised exception —
        the index is rebuilt on the next open regardless, and a supervisor
        that failed to install costs nothing that was not already lost (the
        live session still fires its own wakes).
        """
        await self._transcript.append_custom(
            WAKE_SCHEDULES_CUSTOM_TYPE,
            {"schedules": [schedule.model_dump() for schedule in schedules]},
        )
        self._write_wake_index_entry(schedules, clear=())
        if schedules:
            self._ensure_wake_supervisor()

    def _rebuild_wake_index_entry(self) -> None:
        """Open-time rewrite of the index entry from the scheduler's adopted
        rows. Clears ``stopped_at``: a stopped session's wakes are dormant
        only until someone opens it again, and opening is exactly this."""
        self._write_wake_index_entry(list(self._wake.schedules), clear=("stopped_at",))

    def _write_wake_index_entry(
        self, schedules: list[WakeSchedule], *, clear: tuple[str, ...]
    ) -> None:
        """Best-effort index write. Swallows everything: see
        :meth:`_persist_wake_schedules` for why a failure here must not
        propagate.

        The imports are function-local for the reason every other one in
        this file is: they keep this module's own import cheap and cannot
        form a top-level cycle if ``wakes.store`` ever grows one. The rule
        that matters more — the store never imports the session, because the
        supervisor and the picker read it without a harness — is NOT
        enforced by this; ``tests/unit/test_import_graph.py`` is what pins
        that direction."""
        try:
            from local_operator.paths import config_dir
            from local_operator.wakes import store as wake_store

            root = config_dir()
            # An empty list means "remove the entry", and removal needs
            # nothing from the old file — so skip the read. This is the open
            # path for every session without wakes (including every subagent
            # child), and the directory should not be touched twice for a
            # file that almost never exists.
            existing = wake_store.read_entry(root, self._session_id) if schedules else None
            wake_store.write_entry(
                root,
                self._session_id,
                cwd=self._cwd,
                schedules=schedules,
                preserve=existing,
                clear=clear,
            )
        except Exception:  # noqa: BLE001 — the index is derived; the transcript already has it
            logger.warning("could not update the wake index entry", exc_info=True)

    def _ensure_wake_supervisor(self) -> None:
        """Install-on-demand chokepoint (design §4.2a). Best-effort; the hook
        itself promises never to raise, and this guard is belt-and-braces
        for the same reason the index write has one."""
        try:
            from local_operator.paths import config_dir
            from local_operator.wakes.install import ensure_supervisor_installed

            outcome = ensure_supervisor_installed(config_dir())
            if not outcome.installed:
                logger.debug("wake supervisor not installed: %s", outcome.reason)
        except Exception:  # noqa: BLE001
            logger.warning("wake supervisor install hook failed", exc_info=True)

    # -- subagent roster (resume basis) --------------------------------------

    def _schedule_subagent_persist(self) -> None:
        """Roster-change hook from ``AsyncJobManager`` (synchronous).

        The manager fires this on the hot path of a registration or settle, so
        it must not block or await: it spawns the snapshot write on the session
        task group and returns at once. A child session (one with a ``job_id``)
        does not persist a roster of its own: its parent runner snapshots that
        ledger into the owning row at settlement. Keeping intermediate churn
        off the child transcript avoids quadratic snapshots while the final
        subtree survives durably.
        """
        if self._disposed or self._job_id is not None:
            return
        self._subagent_roster_generation += 1
        if self._subagent_roster_writer is None or self._subagent_roster_writer.done():
            self._subagent_roster_writer = asyncio.create_task(
                self._persist_subagent_roster(), name="persist-subagent-roster"
            )

    async def _persist_subagent_roster(self) -> None:
        """Coalesce current roster state into one atomically replaced sidecar."""
        try:
            while self._subagent_roster_written_generation < self._subagent_roster_generation:
                generation = self._subagent_roster_generation
                rows = [
                    _subagent_job_row(job)
                    for job in self.jobs.list()
                    if getattr(job, "type", "") == "task"
                ]
                records = (
                    self._subagent_comms.snapshot() if self._subagent_comms is not None else []
                )
                compact_records = [_compact_subagent_record(record) for record in records]
                payload = {
                    "version": _SUBAGENT_ROSTER_VERSION,
                    "generation": generation,
                    "jobs": rows,
                    "records": compact_records,
                }
                # The O(roster) fingerprint computation rides the SAME worker
                # hop as the write: #308's invariant is that everything that
                # scales leaves the loop, and an on-loop ``json.dumps`` of a
                # large roster is the exact regression TestMeasurementCosts
                # catches (it fires on every roster event, teardown included).
                fingerprint, wrote = await asyncio.to_thread(
                    _write_roster_sidecar_if_changed,
                    self._transcript.directory / SUBAGENT_ROSTER_SIDECAR,
                    payload,
                    self._persisted_roster_fingerprint,
                )
                if wrote:
                    # Keep one bounded legacy entry for readers predating
                    # v0.35.2; future mutations replace only the sidecar,
                    # avoiding quadratic history while preserving a
                    # rolling-upgrade resume path.
                    if (rows or records) and self._transcript.latest_custom(
                        SUBAGENT_ROSTER_CUSTOM_TYPE
                    ) is None:
                        await self._transcript.append_custom(
                            SUBAGENT_ROSTER_CUSTOM_TYPE,
                            {"jobs": rows, "records": payload["records"]},
                        )
                    self._persisted_roster_fingerprint = fingerprint
                # Advance the written-generation watermark whether or not this
                # payload was actually flushed: the CONTENT is up to date on
                # disk (identical bytes already there), and leaving the watermark
                # behind would make ``_await_subagent_roster_writer`` at teardown
                # loop forever chasing a generation the guard will always skip.
                self._subagent_roster_written_generation = generation
        except Exception:  # noqa: BLE001 - persistence must never break a turn
            logger.warning("could not persist subagent roster", exc_info=True)
        finally:
            if self._subagent_roster_writer is asyncio.current_task():
                self._subagent_roster_writer = None

    async def _await_subagent_roster_writer(self) -> None:
        """Drain the single writer, including a generation queued as it exits."""
        while True:
            writer = self._subagent_roster_writer
            if writer is None:
                if self._subagent_roster_written_generation >= self._subagent_roster_generation:
                    return
                writer = asyncio.create_task(
                    self._persist_subagent_roster(), name="persist-subagent-roster"
                )
                self._subagent_roster_writer = writer
            await asyncio.shield(writer)

    async def _final_persist_snapshots(self) -> None:
        """Write the last roster and todo snapshots at teardown, in order.

        Split out so :meth:`dispose` can bound the pair with a single
        ``wait_for``: both are transcript appends and neither may hang teardown.
        """
        self._subagent_roster_generation += 1
        await self._await_subagent_roster_writer()
        await self._maybe_persist_todos()

    def _load_subagent_roster(self) -> None:
        """Rehydrate the subagent panel and the resume basis from disk.

        Reads the newest snapshot and feeds each half to its owner: the comms
        records to ``SubagentComms.restore`` (the resume basis) and the job rows
        to ``AsyncJobManager.restore`` (the panel). Restoring the records
        MINTS the comms instance if this session had not yet — a resumed session
        that launched children last time is exactly one that needs it — so the
        roster is populated before the first turn can ask for it.

        Malformed rows are dropped individually rather than failing the whole
        load: a resume that lost one child's row is far better than one that
        booted with an empty panel because a single entry was unreadable.

        A restored row has NO in-process page detail: the slim snapshot
        projection (:func:`_subagent_job_row`) drops ``prompt``, ``result_text``
        and ``trajectory``, so opening a restored child's full-page view shows
        an empty body (the view folds those and handles ``None`` without
        raising). That is the accepted tradeoff of not writing a child's full
        output to the parent transcript on every roster move — the detail lives
        in the CHILD's own transcript and is recovered by resuming or
        ``hub op='peek'``-ing it, never by re-reading it here.
        """
        # Forks inherit historical transcript content, not process ownership.
        # The modern sidecar is excluded from the clone, so one present here was
        # written by this fork. For the legacy transcript fallback, compare its
        # append time with the fork boundary so a descendant declines its
        # parent's roster while still restoring one it later writes itself.
        from local_operator.fork import fork_instant

        details = _read_roster_sidecar(self._transcript.directory / SUBAGENT_ROSTER_SIDECAR)
        loaded_sidecar = details is not None
        if details is None:
            entry = self._transcript.latest_custom_entry(SUBAGENT_ROSTER_CUSTOM_TYPE)
            forked_at = fork_instant(self._transcript.directory)
            if entry is None or (forked_at is not None and not entry.ts > forked_at):
                return
            details = dict(entry.payload.get("details", {}))
        if not details:
            return
        self._subagent_roster_generation = int(details.get("generation") or 0)
        self._subagent_roster_written_generation = (
            self._subagent_roster_generation if loaded_sidecar else -1
        )
        records = details.get("records") or []
        logical_by_job: dict[str, str] = {}
        aliases_by_job: dict[str, list[str]] = {}
        if records:
            # ``self.subagent_comms`` (the property) mints the instance on first
            # use; restoring into it is what makes the children addressable.
            try:
                self.subagent_comms.restore(list(records))
                for record in self.subagent_comms.snapshot():
                    job_id = str(record.get("job_id") or "")
                    session_dir = str(record.get("session_dir") or "")
                    if job_id and session_dir:
                        aliases = [
                            str(alias) for alias in record.get("attempt_aliases", []) if alias
                        ]
                        aliases_by_job[job_id] = aliases
                        for attempt_id in [*aliases, job_id]:
                            logical_by_job[attempt_id] = session_dir
            except Exception:  # noqa: BLE001 - a bad snapshot must not stop boot
                logger.warning("could not restore subagent records", exc_info=True)
        rows: list[AsyncJob] = []
        for raw in details.get("jobs") or []:
            try:
                payload = dict(raw)
                job_id = str(payload.get("id") or "")
                # Legacy job snapshots predate ``logical_id``; the comms half
                # already carried the transcript directory, so join the two by
                # attempt id before asking the manager to collapse duplicates.
                payload.setdefault("logical_id", logical_by_job.get(job_id))
                payload.setdefault("attempt_aliases", aliases_by_job.get(job_id, []))
                rows.append(AsyncJob.model_validate(payload))
            except Exception:
                logger.warning("dropping malformed persisted subagent row: %r", raw)
        if rows:
            try:
                self.jobs.restore(rows)
            except Exception:  # noqa: BLE001 - a bad snapshot must not stop boot
                logger.warning("could not restore subagent job rows", exc_info=True)

    # -- todo list (resume) --------------------------------------------------

    async def _maybe_persist_todos(self) -> None:
        """Snapshot the todo list to the transcript when it moved this turn.

        The todo tool keeps the live list in a process-local table, so a
        transcript snapshot is the only durable copy a resume can read. Guarded
        by the FULL-list fingerprint (the same one the continuation guardrail
        compares): an unchanged list is one tuple comparison and no write, while
        any init/add/done/block/drop lands a fresh newest-wins entry. Never
        raises — a status write must not break a turn.
        """
        fingerprint = todo_fingerprint(self._session_id)
        if fingerprint == self._persisted_todo_fingerprint:
            return
        if not fingerprint and self._persisted_todo_fingerprint is None:
            # An empty list that was never persisted is the ordinary
            # never-used-todos session: record nothing and, importantly, do
            # not append an empty snapshot at teardown (which on a wedged mount
            # would charge the dispose budget). The guard above already lets a
            # list that went from populated back to empty through, because its
            # persisted fingerprint is then non-None.
            return
        try:
            await self._transcript.append_custom(
                TODO_SNAPSHOT_CUSTOM_TYPE,
                {"items": todo_snapshot(self._session_id)},
            )
            self._persisted_todo_fingerprint = fingerprint
        except Exception:  # noqa: BLE001 - persistence must never break a turn
            logger.warning("could not persist todo snapshot", exc_info=True)

    def _load_todo_snapshot(self) -> None:
        """Rehydrate the todo list from disk so the panel and guardrail return.

        A child session shares no todo store with its parent (the store is keyed
        by session id), so this is safe for both — a child simply finds no
        snapshot under its own id and does nothing. The persisted fingerprint is
        seeded from the restored list so the first post-resume turn does not
        re-write a list that has not changed.
        """
        details = self._transcript.latest_custom(TODO_SNAPSHOT_CUSTOM_TYPE)
        if not details:
            return
        items = details.get("items") or []
        if items:
            restore_todos(self._session_id, list(items))
            self._persisted_todo_fingerprint = todo_fingerprint(self._session_id)

    # -- active model route (fallback persistence) ---------------------------

    async def _persist_active_route(self, primary: ModelSpec) -> None:
        """Record which model is actually serving requests, for resume.

        Written on EVERY edge, including recovery (``active: None``), because
        ``latest_custom`` reads backwards and stops at the first hit: without
        the recovery row a session that fell back and recovered would resume
        pinned to a fallback nothing is wrong with. ``primary`` rides along so
        the restore can tell a pin that belongs to the CURRENT selected model
        from one stranded by a later ``/model`` switch.
        """
        # The PIN is what persists — the target selector and the chain entry's
        # own effort (None when the entry named none) — not the derived display
        # spec: a restore re-derives against whatever the primary's effort is
        # THEN, exactly as the live failover driver would.
        route = self._active_route
        await self._transcript.append_custom(
            ACTIVE_ROUTE_CUSTOM_TYPE,
            {
                "primary": f"{primary.provider}/{primary.model_id}",
                "active": (None if route is None else {"selector": route[0], "effort": route[1]}),
            },
        )

    # -- selected model (mid-session /model switch persistence) --------------

    def _spawn_selected_model_write(self) -> None:
        """Start (or coalesce onto) the background journal write for the
        selection. The same contract as :meth:`_spawn_conversation_name_write`
        and for the same reason: ``dispose`` cancels ``_background_tasks``
        wholesale, and a switch made in the closing moments of a session is
        exactly the write that must still reach disk — so this task is tracked
        separately and AWAITED by :meth:`_flush_selected_model` instead of
        being cancelled with the rest.

        One task at a time: the payload is read at write time, so a second
        switch landing while a write is in flight is covered by the dirty
        flag — the flush re-persists whatever is in force at teardown.
        """
        if self._disposed:
            return
        task = self._selected_model_task
        if task is not None and not task.done():
            return
        try:
            task = asyncio.ensure_future(self._persist_selected_model())
            # Consume the exception explicitly, as the title's writer does: a
            # failed journal write is tolerated (the flush retries), but a
            # task whose exception nobody reads is reported by the loop at GC
            # time as if the session had leaked it.
            task.add_done_callback(self._on_selected_model_written)
            self._selected_model_task = task
        except RuntimeError:
            # No running loop (a session driven synchronously in tests). The
            # dispose flush is the backstop; the selection is not lost.
            self._selected_model_task = None

    @staticmethod
    def _on_selected_model_written(task: "asyncio.Future[None]") -> None:
        """Retrieve the selection write's outcome so asyncio does not report
        it; the dispose flush is what actually retries a failure."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.debug("model selection write failed; will retry at dispose", exc_info=error)

    async def _persist_selected_model(self) -> None:
        """Journal the model the user is on (newest entry wins on replay).

        A full snapshot per switch, like every other custom-entry journal
        here: ``latest_custom`` scans backward and stops at the first hit, so
        replay cost does not grow with the number of switches. The dirty flag
        is cleared only AFTER the append lands — a write cancelled at teardown
        never reaches that line, so the entry still reads as outstanding to
        :meth:`_flush_selected_model` and is retried there instead of lost.

        A duplicate row (append landed, clear did not) is harmless by
        construction: each row is a snapshot of the same state, and the
        backward scan reads one.
        """
        model = self._model
        payload = {
            "selector": f"{model.provider}/{model.model_id}",
            "effort": model.reasoning_effort,
            "boot": self._boot_selector,
        }
        await self._transcript.append_custom(SELECTED_MODEL_CUSTOM_TYPE, payload)
        current = self._model
        if (f"{current.provider}/{current.model_id}", current.reasoning_effort) == (
            payload["selector"],
            payload["effort"],
        ):
            self._selected_model_dirty = False
        else:
            # The selection moved under the append. Start another pass now
            # rather than leaving the newer selection to the dispose flush:
            # the session can run for hours after a switch, and "correct only
            # if you quit" is not a persistence contract. The current task is
            # still registered until its callback returns, so scheduling onto
            # the next loop tick lets `_spawn_selected_model_write` see it as
            # done and create exactly one successor.
            asyncio.get_running_loop().call_soon(self._spawn_selected_model_write)

    def _restore_selected_model(self) -> None:
        """Re-adopt the model a ``/model`` switch selected before the quit.

        Guarded twice, each a real situation rather than paranoia:

        - persisted ``boot`` differs from THIS construction's boot selection →
          the journal belongs to a boot default the user has since changed (a
          ``/model default`` write, an edited agent profile, an explicit
          ``--hosting``/``--model`` flag on the resume command itself). The
          changed selection is the newer choice and wins; adopting the row
          would make the flag the user just typed silently not work.
        - the journalled selector equals the boot selection → the user
          switched and later switched back; nothing to do, and skipping the
          no-op keeps a fresh session's construction byte-identical to one
          that never switched.

        Tolerant of a malformed or unresolvable entry, matching
        :meth:`_load_conversation_name`: a resume must not be refused because
        one bookkeeping row could not be read, and a selector whose provider
        no longer resolves (an uninstalled registry entry) logs and falls
        back to the boot model rather than constructing a session around a
        spec that cannot serve requests.

        Restores quietly (no event, no journal write): construction runs
        before any front end subscribes, so hosts read ``model`` when they
        build their chrome, and re-journalling the row it just read would
        grow the transcript on every resume.
        """
        details = self._transcript.latest_custom(SELECTED_MODEL_CUSTOM_TYPE)
        if not details:
            return
        selector = str(details.get("selector") or "")
        if "/" not in selector:
            return
        boot = str(details.get("boot") or "")
        if boot != self._boot_selector:
            return
        if selector == self._boot_selector:
            return
        raw_effort = details.get("effort")
        effort = str(raw_effort) if isinstance(raw_effort, str) and raw_effort else None
        # Validate the PROVIDER before adopting the row, exactly as the TUI's
        # `/model` does and for the same reason: `spec_for_target` does not
        # raise on an unknown provider, it returns a spec with `base_url=None`.
        # A provider that has since left the registry (a renamed id, a build
        # without it) would therefore resume the session onto a spec that
        # cannot serve requests, and the failure would surface on the first
        # prompt as a network error rather than as the stale journal row it is.
        from local_operator.providers.registry import get_provider_definition

        provider = selector.split("/", 1)[0]
        if get_provider_definition(provider) is None:
            logger.warning(
                "dropping persisted model selection naming an unknown provider: %r", selector
            )
            return
        # The SAME derivation `/model` itself lands on: `spec_for_target`
        # builds the target model's own spec (its base_url, window,
        # capabilities) and carries only session sampling preferences across
        # — re-deriving instead of persisting the whole spec means a registry
        # update between quit and resume is picked up rather than replayed
        # stale.
        spec = self._spec_for_route(selector, effort)
        if spec is None:
            logger.warning("dropping unresolvable persisted model selection: %r", selector)
            return
        self._model = spec
        # Same synchronous re-fit hook `set_model` ends on. Nothing is frozen
        # this early, but a stream fn that keys caches by selector must start
        # keyed to the model that will actually serve, not the boot default.
        notify = getattr(self._stream_fn, "on_model_changed", None)
        if callable(notify):
            notify(spec)

    async def _flush_selected_model(self) -> None:
        """Land an outstanding selection write before the session tears down.

        The ordinary write is a background task and :meth:`dispose` CANCELS
        background tasks, so a ``/model`` switch in the closing moments of a
        session — the switch-then-ctrl+c this feature exists for — could be
        cancelled before reaching disk, and the next ``--resume`` would open
        on the boot default again. Bounded like the title's flush and for the
        same reason: teardown must not hang on a wedged filesystem, and a
        timeout costs one journal row, never the conversation.
        """
        deadline = time.monotonic() + _NAME_FLUSH_TIMEOUT_S
        task = self._selected_model_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=max(0.0, deadline - time.monotonic())
                )
            except BaseException:  # noqa: BLE001 — fall through to the retry
                pass
        if not self._selected_model_dirty:
            return
        try:
            # Bounded with what REMAINS of the shared deadline, like the
            # title's flush: the append takes the transcript lock, and an
            # unbounded wait here would let a wedged filesystem hold dispose
            # open indefinitely.
            await asyncio.wait_for(
                self._persist_selected_model(),
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except Exception:
            logger.warning("failed to persist the model selection", exc_info=True)

    def _restore_active_route(self) -> None:
        """Re-adopt the fallback that was serving when this session last ran.

        Guarded three ways, each a real situation rather than paranoia:

        - no entry / ``active: None`` → the session closed on its selected
          model (or recovered before closing); nothing to restore.
        - persisted ``primary`` differs from the CURRENT selection → the pin
          belongs to a model the user has since switched away from (a
          ``/model default`` change, an agent profile edit, a ``--hosting``
          flag). The new selection owes the user a fresh start on the model
          they actually chose, not a detour recorded against the old one.
        - the fallback selector equals the current selection → the user
          adopted the fallback as their model; a pin would be a no-op that
          still repainted the band with a spurious fallback marker.

        Restores quietly (no event, no notice): construction runs before any
        front end subscribes, so hosts read ``effective_model`` when they build
        their chrome — the same way they read ``model_label`` — and the replayed
        transcript already narrates the original failure.
        """
        details = self._transcript.latest_custom(ACTIVE_ROUTE_CUSTOM_TYPE)
        if not details:
            return
        active = details.get("active")
        if not isinstance(active, dict):
            return
        selector = str(active.get("selector") or "")
        if "/" not in selector:
            return
        current = f"{self._model.provider}/{self._model.model_id}"
        persisted_primary = str(details.get("primary") or "")
        if persisted_primary and persisted_primary != current:
            return
        if selector == current:
            return
        raw_effort = active.get("effort")
        effort = str(raw_effort) if isinstance(raw_effort, str) and raw_effort else None
        spec = self._spec_for_route(selector, effort)
        if spec is None:
            logger.warning("dropping unresolvable persisted fallback route: %r", selector)
            return
        self._active_fallback = spec
        self._active_route = (selector, effort)
        # Re-pin the stream fn's route too, or the restore is display-only and
        # the first prompt goes back to the provider that was failing. Optional
        # capability like the notice bridge: hosts that construct sessions with
        # bare stream functions simply resume on the primary.
        restore = getattr(self._stream_fn, "restore_fallback", None)
        if callable(restore):
            restore(selector, effort, current)

    async def set_wake_schedules(self, schedules: list[WakeSchedule]) -> None:
        """Full-list update from the wake tool: persists then re-arms."""
        await self._wake.update(schedules)

    async def _deliver_wake(self, due: DueWake) -> None:
        """Deliver one fired wake through the prompt path as a user-attributed
        ``wake_prompt`` custom message. A wake resumed PAST its due time is
        annotated as missed — the agent must not read it as punctual, and a
        recurring one names the skipped occurrences (deduplicated to a count;
        the identical message is NOT repeated per miss)."""
        text = format_wake_delivery_text(due)
        missed_note = self._missed_delivery_note(due)
        if missed_note:
            text = f"{missed_note}\n\n{text}"
        busy = self._is_streaming
        if busy:
            # The resume guidance rides the text itself: the wake reached a
            # turn that was already working, and the message is the only
            # channel that can tell the agent to fold the wake in and then
            # CONTINUE that work. Idle-path deliveries stay clean — they open
            # their own turn, so there is no prior work to resume.
            text = self._append_busy_resume_note(text)
        wake_message = CustomMessage(
            custom_type=WAKE_PROMPT_MESSAGE_TYPE,
            attribution="user",
            details={"wake_id": due.schedule.id, "occurrence": due.occurrence, "text": text},
        )
        # The receipt event rides BEFORE the turn spawn so a front end can
        # paint the expandable wake line ahead of the work it triggered —
        # without it the transcript showed the agent starting to work with no
        # record that a wake was the cause.
        await self._emit(
            WakeDeliveredEvent(
                text=text,
                catchup=False,
                wake_id=due.schedule.id,
                occurrence=due.occurrence,
            )
        )
        if busy:
            # Busy: ride the next successful tool boundary instead of racing
            # the turn — and mark the message COURTESY so the immediate-
            # interrupt poll does not cancel the tool it landed in the middle
            # of (see _has_urgent_steering).
            self._courtesy_wake_count += 1
            self._steering_queue.put_nowait(wake_message)
            # Courtesy toward a MUTATING tool, not toward a parked `wait`. A
            # wake is the user's "remind me", and the agent parked for the
            # very reason the reminder exists; without this mark a wake
            # firing inside a long wait is read only when the budget expires
            # (up to an hour), which is a missed reminder. `wait` returns
            # with the job still running and the wake text lands at the
            # boundary the queue above was already headed for. AFTER the
            # put, for the same lost-wakeup reason receive_peer_message marks
            # last: the woken tool returns into a drain, and the drain must
            # find the message already queued.
            self._peer_arrival.mark(WAKE_PROMPT_MESSAGE_TYPE)
            return
        self._spawn_background(self._prompt_messages([wake_message]))

    @staticmethod
    def _append_busy_resume_note(text: str, *, continue_what: str | None = None) -> str:
        """The busy-path suffix: what to do after the wake's task is handled.

        A wake that lands mid-turn interrupts NOTHING (courtesy delivery), so
        the turn's own work is still owed when the wake is done — the note
        names that obligation, because the alarm envelope alone reads as a
        fresh instruction and 'do the wake task, then go back' is exactly the
        behaviour a wake firing mid-task used to lose. Idle-path deliveries
        stay clean: they open their own turn, so there is no prior work to
        resume.

        ``continue_what`` names the interrupted work when "the work you were
        doing" is not literally true — a catch-up folded ahead of a FRESH
        session's first prompt interrupts nothing yet, so it continues with
        the user's request instead of resuming anything."""
        continuation = continue_what or "resume the work you were doing when it fired"
        return (
            f"{text}\n\n"
            "(This wake fired while you were already working. It was held for a "
            "tool boundary so nothing in flight was interrupted: handle the "
            f"wake's task now, then {continuation} unless this wake makes it "
            "obsolete.)"
        )

    def _missed_delivery_note(self, due: DueWake) -> str | None:
        """The 'this wake fired late' prefix, or None for a punctual fire.

        The two annotate-or-not gates are deliberately DIFFERENT, because they
        answer different questions:

        - Before the resume grace expires, an overdue wake is annotated from
          ``_missed_wake_occurrences`` (the load-time snapshot), because its
          catch-up has not gone out yet and the note is the only place the
          skip is visible.
        - After it, that map is zeroed — the aggregated catch-up prompt now
          carries the count, so re-annotating would tell the agent the same
          skip twice. A LATE one-shot (overdue beyond ``MAX_ARM_MS``, i.e.
          missed while the process was down, not a timer tick late) still
          self-annotates from its own due time, which needs no snapshot.

        The boundary case the asymmetry produces — a one-shot only seconds
        overdue gets a note pre-catch-up and none after — is intentional:
        seconds-overdue is timer jitter, not a "missed while down", and the
        catch-up it would double-report against has by then been delivered."""
        now_ms = int(time.time() * 1000)
        schedule = due.schedule
        occurrences = self._missed_wake_occurrences.get(schedule.id, 0)
        if occurrences and now_ms >= self._resume_grace_ends_ms:
            occurrences = 0  # already aggregated into the catch-up prompt
        if not occurrences:
            # Live-skip detection needs the PRE-fire due time; post-fire it
            # has advanced, so only the resume-time snapshot can say a
            # recurring wake skipped occurrences. A one-shot overdue by more
            # than the timer's resolution is still attributable directly.
            if schedule.every_ms is None and now_ms - schedule.next_due_at > MAX_ARM_MS:
                return (
                    "(This wake was scheduled to fire earlier but is being delivered after "
                    "the session resumed, so it was missed at its scheduled time.)"
                )
            return None
        if occurrences == 1:
            return (
                "(This wake came due while the session was closed and is being delivered "
                "after the resume — 1 occurrence was missed.)"
            )
        return (
            f"(This recurring wake came due while the session was closed and is being "
            f"delivered after the resume — {occurrences} occurrences were missed and have "
            f"been deduplicated into this single delivery.)"
        )

    # -- lifecycle ----------------------------------------------------------------

    def _apply_config_change(self, change: Any) -> None:
        """Apply a ``config.yml`` change to this running session.

        The listener the composition root subscribes to the process's
        :class:`~local_operator.config_watch.ConfigWatcher` (``create_session``
        for top-level sessions, ``_build_child_session`` for subagents). One
        listener per session that fans out to the session's own consumers, so
        the order in which they see a change is deterministic.

        Three groups are live, each for a reason that makes the apply trivial:

        * ``compaction.*`` — re-coerced into a fresh ``CompactionSettings``.
          All three trigger checks read ``self._compaction_settings`` at check
          time, so rebinding it IS the change. A compaction pass already in
          flight keeps the object it captured — a threshold edit does not
          retarget a summary mid-write — and the next check sees the new one.
        * ``retry.*``, ``providers.openai.api``, ``effort.*`` — handed to the
          stream fn's ``apply_settings``, which rebinds the mapping its
          per-call ``RetrySettings.from_settings`` reads. A subagent shares
          the parent's stream fn, so the parent's rebind covers the tree;
          the child still calls it (idempotent) so a child built against a
          parent whose stream fn has no ``apply_settings`` degrades the same
          way as its parent.
        * ``subagents.max_running`` — pushed into the live job manager with
          the same validation the constructor applied; an unset or invalid
          value restores the manager's built-in default rather than freezing
          the last explicit one, so "reset to default" on the page means it.

        Everything else the registry calls LIVE is already read per use
        (``fork.*``, ``web_*`` knobs, ``subagents.models.*``) and needs no
        apply here. Everything it calls NEW_SESSIONS is deliberately ignored.

        Guards on ``_disposed`` because the unsubscribe runs as a dispose
        HOOK, after the session's own teardown, and a tick can land between.
        Swallows nothing else: the watcher isolates a raising listener and
        logs it, and a coercion failure here already degrades to defaults.
        """
        if self._disposed:
            return
        changed = getattr(change, "changed_keys", frozenset())
        values = getattr(change, "values", None)
        if not isinstance(values, Mapping):
            return
        if any(key.startswith("compaction.") for key in changed):
            # Same outcome as the factory's ``coerce_compaction_settings`` at
            # build (dict -> validated, invalid -> defaults, absent -> None,
            # so the read sites' ``or CompactionSettings()`` applies) but via
            # the logging coercion: the factory's prints to stderr, which is
            # the TUI's screen.
            raw = values.get("compaction")
            fresh: Any = (
                _coerce_compaction_settings(dict(raw)) if isinstance(raw, Mapping) else None
            )
            self._compaction_settings = fresh
        # ``effort.*`` is deliberately NOT in this condition (review round 1,
        # M1). The stream fn does read ``self._settings["effort"]`` per message
        # (``configure._effort_for``), but those keys are not in the
        # ``settings_io`` registry, and the diff walks the registry — so
        # ``changed`` can never contain one and a ``startswith("effort.")``
        # clause was unreachable code that read as coverage. An effort edit
        # therefore does not propagate, exactly as before this PR; registering
        # those keys is a separate change with its own scope decision, not
        # something to smuggle in behind a condition nobody could reach.
        #
        # ``providers.`` covers the whole prefix rather than naming
        # ``providers.openai.api`` alone (review round 2, M6). Every key under
        # it is read off the mapping this rebinds — the OpenAI wire surface at
        # client build, the Anthropic cache TTL at the same point — so naming
        # one key meant the OTHER moved only as a side effect of a neighbouring
        # ``retry.*`` edit landing in the same tick, which is both arbitrary
        # and the thing that made its NEW_LAUNCH label a lie. They are all in
        # the LIVE ``providers`` section now, so the trigger matches the scope.
        if any(key.startswith("retry.") or key.startswith("providers.") for key in changed):
            apply_settings = getattr(self._stream_fn, "apply_settings", None)
            if callable(apply_settings):
                apply_settings(values)
        if "subagents.max_running" in changed:
            from local_operator.harness.jobs import DEFAULT_MAX_RUNNING_JOBS

            cap = _configured_max_running(values).get("max_running", DEFAULT_MAX_RUNNING_JOBS)
            try:
                self.jobs.set_max_running(cap)
            except ValueError:
                logger.warning("subagents.max_running=%r rejected by the job manager", cap)

    def add_dispose_hook(self, hook: Callable[[], Awaitable[None] | None]) -> None:
        """Register teardown that runs after the session's own dispose.

        The composition root owns resources the session never created — MCP
        connections, the auth store's SQLite handle, the shared HTTP client —
        and every front end calls ``dispose`` exactly once, so this is the one
        place to hang them. Hooks run in registration order, and one that
        raises is logged rather than propagated: teardown must never mask the
        dispose that triggered it.
        """
        self._dispose_hooks.append(hook)

    async def _close_browser_surface(self) -> None:
        """Close a browser surface the agent left open as a teardown fallback.

        The surface is session-scoped (see ``BrowserSurface``), so nothing else
        can close it after the handle dies with the process. Agents are told to
        close before their final response because interactive TUI/cmux processes
        commonly stay alive between turns; closing on every assistant final
        would instead break legitimate multi-turn browsing and pending login or
        approval flows. Disposal covers process exit, child-runner completion,
        crashes, and missed instruction cleanup for both cmux and bridge tabs.

        The tool layer is imported HERE rather than at module scope: this is a
        teardown-only call, and ``tools.builtin`` is otherwise absent from the
        session's import graph. Failures are logged, never raised — a terminal
        emulator that will not answer must not be able to break dispose.

        Bounded, because ``_run_cmux`` allows a cmux call 30 s and a wedged
        socket is exactly the state a session is likely to be torn down in.
        Cancelling the wait kills the cmux child (``_run_cmux`` handles
        ``CancelledError``), so the timeout does not trade a stranded tab for an
        orphaned process.
        """
        if not self._browser.surface_id:
            return
        try:
            from local_operator.tools.builtin import close_browser_surface

            problem = await asyncio.wait_for(close_browser_surface(self._browser), timeout=5.0)
            if problem:
                logger.warning("could not close browser surface: %s", problem)
        except Exception:
            logger.warning("closing the browser surface failed", exc_info=True)

    async def dispose(self) -> None:
        """Abort any in-flight turn, close the browser surface, cancel
        background work, dispose jobs and the wake scheduler, flush the
        transcript, then run dispose hooks.

        Order matters: the running turn is aborted and AWAITED (bounded)
        before anything is torn down, so its persistence and event emission
        complete against live objects; background wake deliveries and aside
        writes are cancelled next; the task group is closed last.
        """
        if self._disposed:
            return
        self._disposed = True
        # No later boundary can drain producer steers once disposal starts.
        # Reject them while owner callbacks and viewers are still attached so
        # capacity is released and the producer can safely reuse the same ID.
        while not self._steering_queue.empty():
            message = self._steering_queue.get_nowait()
            command_id = self._steering_producers.pop(id(message), None)
            if command_id is not None:
                await self._reject_steering(command_id, "session closed before durable admission")
        # A courtesy wake still queued here was never delivered, so its count
        # must not survive to misclassify a later enqueue on a reused Session.
        self._courtesy_wake_count = 0
        try:
            # HC-14: abort the in-flight turn and await its completion (bounded)
            # before flushing — its persistence must land on a live transcript.
            turn = self._turn_task
            if turn is not None and not turn.done() and self._signal is not None:
                self.abort("session disposed")
                try:
                    await asyncio.wait_for(asyncio.shield(turn), timeout=5.0)
                except BaseException:  # noqa: BLE001 — dispose must always proceed
                    pass
            # The browser surface is session-scoped and lives in the user's own
            # browser, so an unclosed one is a tab THEY have to close by hand.
            # After the turn has stopped (so nothing is mid-navigation on it)
            # and before the task group closes, since this awaits a subprocess.
            await self._close_browser_surface()
            # BEFORE the background tasks are cancelled: the title's own write
            # is one of them, and a name stored in the last moments of the
            # session (a `/rename` before ctrl+d) would be cancelled in flight
            # and never reach the transcript the next `--resume` reads.
            await self._flush_conversation_name()
            # Same window, same reason: the selection journal's ordinary write
            # is a background task about to be cancelled, and a `/model`
            # switch made moments before quitting is exactly the write that
            # must still land for the next `--resume` to honour it.
            await self._flush_selected_model()
            # Fold any still-parked journal notice back into the live context
            # before the durability flush reads it. The transcript write
            # already happened at park time, so `--resume` is correct either
            # way and `_persist_new_messages` dedups by id; this keeps the
            # in-memory list consistent with what was persisted rather than
            # leaving a notice stranded in the FIFO at teardown.
            self._flush_context_journal()
            # HC-11: cancel tracked background tasks (wake deliveries, aside
            # persistence), then close the session task group.
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            if self._tg_stack is not None:
                with contextlib.suppress(Exception):
                    await self._tg_stack.aclose()
                self._tg_stack = None
            self._task_group = None
            # Snapshot the roster and todos ONE last time, directly (the task
            # group is closed, so ``_spawn_background`` no longer runs) and
            # BEFORE ``jobs.dispose`` cancels the running children — a child
            # cancelled by teardown settles ``cancelled``, and persisting after
            # that would record a status the resume then offers to "resume"
            # from a run the user only stopped by quitting. Persisting first
            # captures each child as it actually stood at quit. Guarded so a
            # write failure never blocks the rest of teardown. Skipped for a
            # child session (it persists no roster of its own).
            if self._job_id is None:
                try:
                    # Bounded like the conversation-name flush above: a snapshot
                    # is a transcript append, and teardown must not hang on a
                    # wedged mount or behind a lock a stuck writer holds. A lost
                    # final snapshot is cheap — the per-turn and roster-change
                    # snapshots already captured all but the last few
                    # milliseconds — so the deadline favours proceeding.
                    await asyncio.wait_for(
                        self._final_persist_snapshots(), timeout=_NAME_FLUSH_TIMEOUT_S
                    )
                except (Exception, asyncio.TimeoutError):  # noqa: BLE001
                    logger.warning("final roster/todo snapshot did not land", exc_info=True)
            await self.jobs.dispose()
            self._wake.dispose()
            # A shell receipt can be queued behind a turn that was just aborted.
            # Its normal turn-finally flush should already have run, but this
            # last-resort pass keeps disposal durable if teardown reached here
            # through a host path without a turn task.
            if self._pending_shell_records:
                await self._flush_shell_records()
            self._transcript.flush()
        finally:
            # Drop the retention claim FIRST in the finally: everything in the
            # try above can raise, and a dispose that blew up part way through
            # must not leave the directory's liveness marker behind for the rest
            # of the process's life. On POSIX a leaked marker heals when the pid
            # dies, but releasing it promptly means an EMPTY session directory
            # (a run that wrote nothing) becomes reapable the moment its run
            # ends rather than when the OS happens to reuse the pid — and matters
            # most for a host running several sessions in one long-lived process.
            # A directory with real content is untouchable regardless, so this
            # never risks the transcript itself. ``release_session`` is a no-op
            # for agent directories (the gate lives inside it).
            from local_operator.session.retention import release_session

            release_session(self._transcript.directory)
            # ``finally``: host-owned resources must be released even when the
            # session's own teardown blew up part way through.
            for hook in self._dispose_hooks:
                try:
                    outcome = hook()
                    if inspect.isawaitable(outcome):
                        await outcome
                except Exception:
                    logger.warning("session dispose hook failed", exc_info=True)
