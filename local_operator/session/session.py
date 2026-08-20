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
  ``steer()`` is the public injection point.
- Wakes persist as a ``wake_schedules`` custom transcript entry (newest wins)
  and deliver through the prompt path as a user-attributed
  ``wake_prompt`` custom message.
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
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from local_operator.compaction.tokens import IMAGE_TOKEN_ESTIMATE, approx_text_tokens
from local_operator.harness.approval import ApprovalGate
from local_operator.harness.comms import HUB_MESSAGE_TYPE, SubagentComms
from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE, AsyncJobManager
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
    ModelChangeEvent,
    ModelSpec,
    NoticeEvent,
    StaleAside,
    SteeringDeliveredEvent,
    StreamEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    ToolContext,
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
from local_operator.incidents import SESSION_INCIDENT_MESSAGE_TYPE
from local_operator.session.goal import GoalState
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.naming import (
    CONVERSATION_NAME_CUSTOM_TYPE,
    MAX_TITLE_CHARS,
    ConversationName,
)
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import (
    TODO_REMINDER_MESSAGE_TYPE,
    open_todos,
    todo_fingerprint,
)

if TYPE_CHECKING:
    # Type-only: the session must never pull the MCP stack in at import time.
    # It only holds the manager the composition root hands it.
    from local_operator.mcp.manager import McpManager

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


def _configured_max_running() -> dict[str, int]:
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

    Never raises: a malformed config must not stop a session from starting.
    A non-positive value is rejected rather than honoured, because 0 would
    deadlock every launch behind a gate that can never open.
    """
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        raw = ConfigManager(config_dir()).get_config_value("subagents", None)
        if not isinstance(raw, dict):
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


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default transcript→LLM rendering.

    ``compaction_summary`` markers become a user message carrying the summary;
    a snapcompact archive in ``preserve_data`` is rendered back into
    text_head → imaged middle → text_tail blocks (base64 ``ImageContent``
    between ``TextContent`` edges). ``wake_prompt`` deliveries become user
    messages of their formatted text, and the newest ``todo_reminder`` (only
    the newest) becomes one too; other custom entries are dropped (bookkeeping
    never enters LLM context). ``provider_payload`` rides along untouched.
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
        elif message.custom_type == SESSION_INCIDENT_MESSAGE_TYPE:
            # An incident rides the sender's preformatted text (the classifier
            # already wrote category + suggested action), exactly like a wake
            # delivery: it must reach the model as a user turn or the session
            # stays blind to why its last run died.
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
        elif message.custom_type in (
            WAKE_PROMPT_MESSAGE_TYPE,
            HUB_MESSAGE_TYPE,
            JOB_RESULT_MESSAGE_TYPE,
        ):
            # A hub message renders exactly like a wake delivery: the sender
            # already formatted ``details["text"]``, and it must reach the
            # model as a user turn or the agent it was addressed to never
            # sees it. Unlisted custom types are dropped (bookkeeping), which
            # is precisely the trap a new aside type falls into.
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
_PERSISTABLE_CUSTOM_TYPES: frozenset[str] = frozenset({SESSION_INCIDENT_MESSAGE_TYPE})


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

    :meth:`Session._history_snapshot` faces the same illegal tail for a
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
    """
    if isinstance(message, CustomMessage):
        return message.custom_type in _PERSISTABLE_CUSTOM_TYPES
    return True


def _stamped_todo_fingerprint(details: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The todo fingerprint a reminder was built from, normalized for compare.

    Read back through an explicit tuple/str coercion rather than trusted as
    stored: ``details`` is a plain dict, and any JSON round trip turns the
    nested tuples into lists — a raw ``!=`` would then be True for an unchanged
    list and expire every reminder on sight. A reminder with no stamp (or a
    malformed one) compares equal to nothing and expires, which is the safe
    direction: a nudge that may be lying is worth less than one turn without it.
    """
    stamped = details.get("fingerprint") or ()
    return tuple(
        (str(item[0]), str(item[1]))
        for item in stamped
        if isinstance(item, (list, tuple)) and len(item) == 2
    )


def _replayed_user_message(content: list[Content], entry_id: str | None) -> Message:
    """Build a replayed user message, preserving its transcript entry id.

    A message rendered from a persisted entry MUST keep that entry's id:
    ``first_kept_entry_id`` references it, so minting a fresh uuid here would
    make replay unable to find the cut point. A message with no originating
    entry keeps the model's default id.
    """
    message = Message(role="user", content=content)
    if entry_id:
        message.id = entry_id
    return message


#: Stands in for an image the provider refused, so the turn that follows it
#: still makes sense. A silently shortened message would leave the model
#: reading a summary whose "the screenshots below" no longer has any below.
IMAGE_DROPPED_NOTICE = "[image omitted: the provider rejected it and it has been dropped]"


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


def _without_images(messages: list[Message]) -> list[Message]:
    """Every message with its image blocks replaced by a one-line notice.

    Used after a provider has refused an image (see
    :func:`~local_operator.providers.failover.is_image_rejection`). Applied to
    the RENDERED history rather than to the transcript, so nothing is destroyed:
    the archive keeps its frames, ``/export`` still has them, and a later
    session on a provider that accepts them is unaffected.

    Consecutive images collapse to ONE notice. A snapcompact archive replays as
    fifty-odd frames between two text edges, and fifty identical apology lines
    would cost more context than the summary they are standing in for.
    """
    out: list[Message] = []
    for message in messages:
        if not any(isinstance(block, ImageContent) for block in message.content):
            out.append(message)
            continue
        content: list[Content] = []
        for block in message.content:
            if isinstance(block, ImageContent):
                if content and getattr(content[-1], "text", None) == IMAGE_DROPPED_NOTICE:
                    continue
                content.append(TextContent(text=IMAGE_DROPPED_NOTICE))
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


def _render_compaction_marker(marker: CustomMessage, entry_id: str | None = None) -> Message:
    """Render one compaction marker into an LLM-visible message. ``entry_id``
    (the marker's transcript entry id) rides onto the rendered message.

    Snapcompact archives replay via ``history_blocks`` (lazy import; any
    failure degrades to the plain-text summary so a malformed archive never
    breaks the turn).
    """
    summary = marker.details.get("summary", "")
    preserve = marker.details.get("preserve_data") or {}
    archive_payload = preserve.get("snapcompact")
    if archive_payload:
        try:
            from local_operator.compaction import snapcompact

            archive = snapcompact.Archive.model_validate(archive_payload)
            content: list[TextContent | ImageContent] = []
            for block in snapcompact.history_blocks(archive):
                if block["kind"] == "text":
                    content.append(TextContent(text=block["text"]))
                elif block["kind"] == "images":
                    for frame_b64 in block["frames"]:
                        content.append(ImageContent(data=frame_b64, mime_type="image/png"))
            if content:
                return _replayed_user_message(content, entry_id)
        except Exception:
            logger.warning("snapcompact replay failed; falling back to text summary", exc_info=True)
    # A snapcompact summary is reading instructions for the frames, not a
    # digest of the history — falling back to it ALONE would replay a caption
    # describing images that are not there while the real content vanished.
    # The archive's text edges are plain strings in the same payload and
    # survive whatever made the frame list unrevivable, so salvage them: they
    # are the newest/oldest slices of the actual transcript, which is strictly
    # more useful than any caption.
    salvage = ""
    if isinstance(archive_payload, dict):
        head = archive_payload.get("text_head")
        tail = archive_payload.get("text_tail")
        edges = [edge for edge in (head, tail) if isinstance(edge, str) and edge.strip()]
        if edges:
            joined = "\n[...]\n".join(edges)
            # The summary above may describe pixel-font frames; none are in
            # this message, and a caption describing absent images is a claim
            # the model would waste attention reconciling. Say so explicitly.
            salvage = (
                "\n[note: the archive's image frames could not be replayed here; "
                "the plain-text edges below are what survives]"
                f"\n<archived-transcript-edges>\n{joined}\n</archived-transcript-edges>"
            )
    return _replayed_user_message(
        [
            TextContent(
                text="<previous-context-summary>\n"
                f"{summary}\n"
                "</previous-context-summary>"
                f"{salvage}"
            )
        ],
        entry_id,
    )


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
        conversation_name: ConversationName | None = None,
        system_blocks_provider: Callable[[], list[str]] | Callable[[], Awaitable[list[str]]],
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
        self._tools = list(tools)
        self._transcript = transcript
        self._session_id = session_id or transcript.directory.name
        self._agent_id = agent_id
        # The goal rides the prompt's volatile tail; the holder is shared with
        # the system-blocks provider so an edit applies from the next turn.
        self._goal_state = goal_state if goal_state is not None else GoalState()
        self._variables = variables
        self._job_id = job_id
        self._subagent_comms = subagent_comms
        self.agent_registry = agent_registry
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
        self._system_blocks_provider = system_blocks_provider
        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        #: Set once a provider refuses a request because of an image block, and
        #: never cleared: from then on this session renders its history without
        #: images. See :func:`~local_operator.providers.failover.is_image_rejection`
        #: for why recovery has to be sticky rather than per-request — the
        #: offending block is IN the history, so an un-degraded retry sends it
        #: again, and the session is otherwise unusable for good.
        self._images_rejected = False
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
        self._context = LoopContext(
            system_blocks=[],
            messages=list(transcript.build_llm_history()),
            tools=self._tools,
        )
        self._handlers: list[EventHandler] = []
        self._steering_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
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
        self._last_activity_ms: int = 0  # epoch ms; drives idle-flush pruning
        self._generation = 0  # monotonic turn counter for agent_start/end
        # Boundary-event suppression across a post-compaction continuation:
        # `_held_end` parks the loop's agent_end until compaction has decided
        # whether the run continues, `_logical_generation` remembers which
        # agent_start the eventual end belongs to. Both are None outside a run.
        self._held_end: AgentEndEvent | None = None
        self._abort_requested = False  # sticky across the continuation gap
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
        self._todo_reminder_fingerprint: tuple[tuple[str, str], ...] | None = None

        self._disposed = False
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
        self._turn_lock = asyncio.Lock()  # serializes prompt() and wake deliveries
        # True only while an ON-DEMAND compaction holds ``_turn_lock``. The
        # automatic pass runs inside a turn and is covered by ``_is_streaming``;
        # this one runs between turns, so a caller that finds the lock held
        # needs a way to tell WHICH holder it is looking at (see `prompt`).
        self._compacting = False
        # Error text from the run just ended, journalled as a session incident
        # once persistence finishes (see _run_turn / journal_incident).
        self._pending_incident: str | None = None
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
        self.jobs = AsyncJobManager(
            on_job_complete=self._on_job_completed,
            **_configured_max_running(),
        )
        self._wake = WakeScheduler(
            now=lambda: int(time.time() * 1000),
            # Deliveries route through the indirection hook: at resume the
            # session swaps in a catch-up shim that folds N overdue per-schedule
            # fires into one aggregated prompt (see _handle_missed_wakes).
            deliver=lambda due: self._wake_deliver_via_hook(due),
            persist=self._persist_wake_schedules,
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
        self._load_wake_schedules()
        self._prepare_missed_wake_catchup()
        # The conversation's title is restored from the SAME transcript the
        # history came from, and for the same reason: a resumed session is the
        # same conversation, so the band and the terminal tab must name it the
        # way it was named before. Loaded here rather than by the host because
        # the transcript is the session's to read — every front end resuming a
        # session would otherwise need its own copy of this, and the one that
        # forgot would boot nameless.
        self._load_conversation_name()
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

    def _render_history(self, messages: list[AgentMessage]) -> list[Message]:
        """The configured transcript→LLM conversion, minus anything a provider
        has already refused.

        Every path that builds wire history goes through here rather than
        calling ``_convert_to_llm`` directly, because the degrade has to hold
        for ALL of them. Compaction is the one that matters most: it has to
        send the history to summarise it, so a poisoned block makes even the
        escape hatch fail (anthropics/claude-code#50708).

        Expired todo reminders are dropped here for the same reason: every path
        that reaches a provider has to be free of them.
        """
        rendered = self._convert_to_llm(self._live_todo_reminders(messages))
        if self._images_rejected:
            # Nothing to rebound once images are being dropped outright, and
            # dropping first saves decoding a block that is about to become a
            # one-line notice.
            return _without_images(rendered)
        return _rebound_history_images(rendered)

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

    def _render_for_compaction(self) -> list[Message]:
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
        """
        return self._render_history(
            [message for message in self._context.messages if not _is_todo_reminder(message)]
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

    @property
    def model(self) -> ModelSpec:
        """The spec every provider call is built from."""
        return self._model

    @property
    def active_fallback(self) -> ModelSpec | None:
        """The pinned fallback's spec while one is serving requests, else None."""
        return self._active_fallback

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

    def set_model(self, model: ModelSpec) -> None:
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
        """
        previous = self._model
        self._model = model
        if (previous.provider, previous.model_id) == (model.provider, model.model_id):
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
            return
        # An explicit switch withdraws the fallback pin's premise: the pin
        # rescued the PREVIOUS selection, and the stream fn's preflight will
        # clear its own route state the moment it sees the new selector. The
        # display state has to move in the same step — a band still naming the
        # old fallback after `/model` is the same stale frame this state
        # exists to prevent. Persisted (with the new primary) so a resume does
        # not restore a pin the user already switched away from.
        if self._active_fallback is not None:
            self._active_fallback = None
            self._active_route = None
            self._spawn_background(self._persist_active_route(model))
            self._spawn_background(
                self._emit(
                    ModelChangeEvent(
                        provider=model.provider,
                        model_id=model.model_id,
                        effort=model.reasoning_effort,
                        reason="model switched",
                        is_fallback=False,
                        context_window=int(model.context_window),
                    )
                )
            )
        notify = getattr(self._stream_fn, "on_model_changed", None)
        if callable(notify):
            notify(model)

    @property
    def goal(self) -> str:
        """The session's standing objective ("" when unset)."""
        return self._goal_state.text

    def set_goal(self, text: str) -> str:
        """Set (or clear, with an empty string) the standing objective.

        Returns what was actually stored (trimmed and length-capped). The
        goal rides the system prompt's volatile tail, so it applies from the
        next turn and only invalidates that tail — never the cached prefix.
        """
        return self._goal_state.set(text)

    @property
    def conversation_name(self) -> str:
        """The conversation's title ("" until one is set or generated)."""
        return self._conversation_name.text

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
            self._conversation_name_dirty = True
            self._spawn_conversation_name_write()
        return stored

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

    async def preflight_usage(self) -> None:
        """Run the stream's message-boundary quota check without starting a turn.

        The TUI calls this after subscribing, so an exhausted default provider
        becomes a visible warning while startup itself remains successful.
        """
        preflight = getattr(self._stream_fn, "preflight_usage", None)
        if not callable(preflight):
            return
        result = preflight(self._model)
        if inspect.isawaitable(result):
            await result

    # -- driving turns --------------------------------------------------------
    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        """Run one user turn to completion (awaitable) or raise.

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
            # A fresh user prompt supersedes any earlier interrupt request.
            self._abort_requested = False
            if self._is_streaming:
                raise RuntimeError("session is already streaming; use steer() to inject mid-turn")
            # Flush a pending wake catch-up before the user's own turn: on a
            # session built inside a running loop (the TUI) this prompt is the
            # only trigger guaranteed to exist, and the aggregated missed
            # wakes belong ahead of the work they were meant to start.
            self._handle_missed_wakes()
            await self._run_turn_pipeline([Message.user(text, images)])
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

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        """Inject a steering message into the running turn (interrupts tool
        batches at the next boundary).

        Attachments ride along for the same reason they do on ``prompt``:
        steering mid-turn with a screenshot is the case where the picture
        IS the correction."""
        self._steering_queue.put_nowait(Message.user(text, images))

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
        """
        self._abort_requested = True
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

    def set_fallback_tool_resolver(
        self, resolver: Callable[[str], AgentTool | None] | None
    ) -> None:
        """Install a resolver for tool names NOT in the inventory (deferred /
        lazy MCP tools). Wired to ``LoopConfig.resolve_fallback_tool`` so the
        loop can dispatch calls to tools not yet materialized. ``None`` clears
        it."""
        self._fallback_tool_resolver = resolver

    # -- context accounting ---------------------------------------------------

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
        blocks = self._system_blocks_provider()
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
            # Same flush as prompt(): a wake-delivery turn is still a prompt
            # path, and the catch-up must not be stranded behind it.
            self._handle_missed_wakes()
            await self._run_turn_pipeline(initial)

    async def _run_turn_pipeline(self, initial: list[AgentMessage]) -> None:
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
            await self._run_turn(initial)
            await self._drain_continuation()
        finally:
            self._turn_task = None
            await self._flush_held_end()

    async def _flush_held_end(self) -> None:
        """Emit the boundary event the pipeline was holding, if any."""
        held = self._held_end
        self._held_end = None
        if held is None:
            return
        generation = self._logical_generation or held.generation
        self._logical_generation = None
        await self._emit(
            held
            if held.generation == generation
            else held.model_copy(update={"generation": generation})
        )

    async def _run_turn(self, initial: list[AgentMessage]) -> None:
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
                await self._transcript.append_message(message)

            blocks = self._system_blocks_provider()
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
                convert_to_llm=self._render_history,
                stream_fn=self._stream_fn,
                get_steering_messages=self._drain_steering,
                has_steering_messages=lambda: not self._steering_queue.empty(),
                get_aside_messages=self._drain_asides,
                get_follow_up_messages=self._todo_continuation,
                resolve_fallback_tool=self._fallback_tool_resolver,
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
                await self._emit(event)

            # Track the latest provider usage for compaction trigger math.
            for message in reversed(new_messages):
                if isinstance(message, Message) and message.usage is not None:
                    self._last_usage = message.usage
                    break
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

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Route a fire-and-forget coroutine through the session task group
        when one is open (wake deliveries, aside persistence); otherwise fall
        back to ``ensure_future``. Every spawned task is tracked so
        :meth:`dispose` can cancel and await it. After dispose nothing is
        spawned, so a late wake delivery cannot raise into an unobserved task.

        The coroutine is wrapped so its failure is logged, never raised: an
        exception escaping into a TaskGroup would cancel every sibling task.
        """
        if self._disposed:
            return

        async def _guarded() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("background session task failed", exc_info=True)

        if self._task_group is not None:
            task = self._task_group.create_task(_guarded())
        else:
            task = asyncio.ensure_future(_guarded())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
            agent_id=self._agent_id,
            has_ui=self._has_ui,
            resolve_internal_url=self._skill_resolver,
            request_approval=None if self._yolo else self._request_approval,
            ask_user=self._ask_user,
            wake_scheduler=self._wake,
            browser=self._browser,
            subagent_launcher=self._launch_subagent,
            jobs=self.jobs,
            subagent_comms=self.subagent_comms,
            variables=self._variables,
            job_id=self._job_id,
            agent_registry=self.agent_registry,
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
        )

    def _resolve_subagent_model(self, agent: str, effort: str | None) -> ModelSpec | None:
        """Effort tier -> ModelSpec via config; None keeps the parent's model.

        Precedence: an explicit ``effort`` on the launch beats the role's own
        default, which beats the session model. That order is what lets a role
        say "this job is usually cheap" while a caller who knows this instance
        is hard can still pay for the better model.
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
        if wanted is None and agent == "scout":
            wanted = "lo"
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
            self._context.messages.append(message)
        except OSError:
            logger.warning("could not journal session incident", exc_info=True)

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
        messages: list[AgentMessage] = []
        while not self._steering_queue.empty():
            message = self._steering_queue.get_nowait()
            await self._transcript.append_message(message)
            messages.append(message)
        if messages:
            # After persistence, not before: the receipt says the message is in
            # the conversation, and it is only in the conversation once it is on
            # disk and in the list being handed back to the loop.
            await self._emit(SteeringDeliveredEvent(count=len(messages)))
        return messages

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
        # The post-run usage scan has not happened yet — the boundary
        # snapshot carries the assistant message that just finished, whose
        # usage is the provider's ground truth for the trigger math.
        for message in reversed(messages):
            if isinstance(message, Message) and message.usage is not None:
                self._last_usage = message.usage
                break
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
        if provider_reported is not None:
            from local_operator.compaction import api as compaction_api

            if not compaction_api.should_compact(
                provider_reported, self.effective_model.context_window, settings
            ):
                return None
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
            return None
        outcome = await self._run_compaction(planned, reason="mid-turn")
        if not outcome.ran:
            return None
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
        planned = await self._plan_compaction(respect_threshold=True)
        if isinstance(planned, CompactionOutcome):
            # Refused: below threshold, disabled, nothing worth summarizing.
            # The automatic path has nobody to tell — a turn that did not need
            # compacting must not narrate that fact every time.
            return
        outcome = await self._run_compaction(planned, reason="context-window")
        if not outcome.ran:
            return

        # (5) Recovery band: only schedule a continuation when the pass
        # actually created headroom (an anti-thrash guard).
        if getattr(planned.settings, "auto_continue", False):
            compaction_api = planned.compaction_api
            threshold = compaction_api.resolve_threshold_tokens(
                self.effective_model.context_window, planned.settings
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
            if not compaction_api.should_compact(
                compaction_api.compaction_context_tokens(provider_reported, bound),
                self.effective_model.context_window,
                settings,
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
        if respect_threshold and not compaction_api.should_compact(
            context_tokens, self.effective_model.context_window, settings
        ):
            return CompactionOutcome(ran=False, reason="below_threshold")

        cut = await self._offloaded(
            compaction_api, "find_cut_point", llm_history, settings.keep_recent_tokens
        )
        if cut is None or cut <= 0:
            # ``find_cut_point`` is the ONE definition of "worth summarizing":
            # the kept window has to reach ``keep_recent_tokens`` and at least
            # two real messages have to fall outside it. Both states a manual
            # trigger runs into land here — a context too small to have older
            # history, and a context whose older history a previous pass has
            # already summarized — so the refusal names which one it is rather
            # than guessing.
            keep_recent = settings.keep_recent_tokens
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
        )

    async def _run_compaction(self, plan: _CompactionPlan, *, reason: str) -> CompactionOutcome:
        """Commit one compaction pass — THE pass, for both triggers.

        ``reason`` rides the two events so a host can tell an automatic pass
        from one the user asked for while keeping one vocabulary for both
        (``compacting context…`` / ``context compacted``).
        """
        compaction_api = plan.compaction_api
        await self._emit(CompactionStartEvent(reason=reason))
        try:
            to_summarize = plan.llm_history[: plan.cut]
            kept = plan.llm_history[plan.cut :]
            summary, preserve_data = await self._produce_summary(
                compaction_api, to_summarize, plan.strategy
            )
            first_kept_entry_id = kept[0].id
            await self._transcript.append_compaction(
                summary, first_kept_entry_id, plan.context_tokens, preserve_data=preserve_data
            )
            marker_details: dict[str, Any] = {"summary": summary}
            if preserve_data is not None:
                marker_details["preserve_data"] = preserve_data
            marker = CustomMessage(
                custom_type="compaction_summary",
                attribution="system",
                details=marker_details,
            )
            # The context becomes the RENDERED history, so a live todo reminder
            # does not survive this — by design, and the reason the plan renders
            # without them (see :meth:`_render_for_compaction`): a reminder
            # baked in here as a plain user message is past both of the guards
            # that expire it.
            self._context.messages = [marker, *kept]
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
            snap_payload = (preserve_data or {}).get("snapcompact")
            if isinstance(snap_payload, dict):
                try:
                    from local_operator.compaction.snapcompact import (
                        frame_token_estimate_for,
                    )

                    frame_count = len(snap_payload.get("frames") or [])
                    per_frame = frame_token_estimate_for(self._model.provider, self._model.model_id)
                    history_after += frame_count * (per_frame - IMAGE_TOKEN_ESTIMATE)
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
            # But the SAVING is measured with one ruler. The provider figure is
            # the full request (system blocks + tool schemas + history) while
            # the local estimates cover history alone, so ``context_tokens -
            # history_after`` would count the fixed overhead as if the pass had
            # removed it — inflating the receipt's "% smaller" and collapsing
            # the band (which subtracts before-after) to a figure that
            # understates the next request by the whole overhead. The honest
            # after-figure keeps the overhead on both sides:
            # ``context_tokens - (history saving)``, floored at the history
            # itself for the degenerate case where the plan's figures cross.
            saved = max(0, plan.tokens_before - history_after)
            tokens_after = max(history_after, plan.context_tokens - saved)
            await self._emit(
                CompactionEndEvent(
                    reason=reason,
                    success=True,
                    strategy=plan.strategy,
                    tokens_before=plan.context_tokens,
                    tokens_after=tokens_after,
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
    #: that ignores the output format can bill us for. Not tight, deliberately:
    #: the cap counts EVERY response token, and on a provider with adaptive
    #: thinking (Anthropic's ``thinking: adaptive``, which this session sends
    #: whenever the model has an effort ladder) some of them are thinking. Seven
    #: measured naming calls emitted 18–38 tokens, so a 64-token cap was running
    #: at 60% of budget on a title we would then have to REJECT for being
    #: truncated. Only the tokens produced are billed, so headroom is free.
    ERRAND_MAX_TOKENS = 128

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
          128-token cap counts thinking tokens as well as the title.
        """
        model = self._errand_model()
        request = ChatRequest(
            model=model,
            system_blocks=[system],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
            max_tokens=self.ERRAND_MAX_TOKENS,
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
        With no tier configured it falls back to the session's own model.

        The clamp is applied to WHICHEVER of the two this returns, and it is not
        only a cost argument. ``ERRAND_MAX_TOKENS`` becomes
        ``max_output_tokens``, which counts reasoning tokens too, so a spec left
        on its provider's default effort can spend the whole 128-token budget
        thinking, emit no ``<title>`` at all and make ``parse_title`` return
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
        return self._lowest_effort(tier if tier is not None else self._model)

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

        No tools, and ``tool_choice="none"``: an aside that could edit a file
        would be a turn wearing a popup, and the answer is meant to come from
        context the agent already has.

        Safe to call mid-turn, and the pairing below is what makes that true —
        see :meth:`_wire_legal_snapshot`.
        """
        blocks = self._system_blocks_provider()
        if inspect.isawaitable(blocks):
            blocks = await blocks
        request = ChatRequest(
            model=self._model,
            system_blocks=list(blocks),
            messages=self._render_history([*self._wire_legal_snapshot(), *turns]),
            tools=[],
            tool_choice="none",
        )
        parts: list[str] = []
        async for event in self._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
                if on_delta is not None:
                    on_delta(event.delta)
            elif isinstance(event, StreamUsageEvent) and on_usage is not None:
                on_usage(event.usage)
        return "".join(parts)

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

    # -- wakes -------------------------------------------------------------------

    def _load_wake_schedules(self) -> None:
        details = self._transcript.latest_custom(WAKE_SCHEDULES_CUSTOM_TYPE)
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
        self._resume_catchup_text = self._format_missed_wake_catchup(missed, now)
        # Install the suppression shim NOW, not at first trigger: it is cheap
        # (a no-op passthrough once the catch-up is sent) and installing it
        # late is what let the scheduler's own tick swallow a per-schedule
        # delivery before any trigger had run — the wake then reached the
        # model nowhere (review round 2, M1).
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
        if self._resume_catchup_sent or self._resume_catchup_text is None:
            return
        if int(time.time() * 1000) < self._resume_grace_ends_ms:
            return
        text, self._resume_catchup_text = self._resume_catchup_text, None
        self._resume_catchup_sent = True
        catchup = CustomMessage(
            custom_type=WAKE_PROMPT_MESSAGE_TYPE,
            attribution="user",
            details={"wake_catchup": True, "text": text},
        )
        # The receipt is a CATCH-UP marker: the front end renders one
        # expandable line for the whole folded set, and the replay loop skips
        # re-adding it as a user row (``wake_catchup``), so this event is the
        # only place the missed wakes are ever surfaced.
        self._emit_nowait(WakeDeliveredEvent(text=text, catchup=True))
        if self._is_streaming:
            self._steering_queue.put_nowait(catchup)
            return
        self._spawn_background(self._prompt_messages([catchup]))

    async def _deliver_wake_catchup(self, due: DueWake) -> None:
        """Deliver hook while the resume catch-up is pending: swallow the
        scheduler's per-schedule fire (the aggregated prompt covers it, and
        the schedule is still advanced + persisted by pump), and fall back to
        a normal annotated delivery for any wake that becomes due AFTER the
        catch-up went out — grace already expired for it, so a direct fire
        with the missed-occurrence note is correct, not a duplicate."""
        if self._resume_catchup_sent:
            await self._deliver_wake(due)
        # Pending: swallow — the aggregated prompt covers this fire. pump()
        # still advances and persists the schedule, so nothing re-fires.

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
        await self._transcript.append_custom(
            WAKE_SCHEDULES_CUSTOM_TYPE,
            {"schedules": [schedule.model_dump() for schedule in schedules]},
        )

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
        if self._is_streaming:
            # Busy: ride the next steering boundary instead of racing the turn.
            self._steering_queue.put_nowait(wake_message)
            return
        self._spawn_background(self._prompt_messages([wake_message]))

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
        """Close a cmux browser surface the agent left open.

        The surface is session-scoped (see ``BrowserSurface``), so nothing else
        will ever close it: the model is not guaranteed to call ``browser
        close``, and the handle dies with the process.

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
            # cmux pane, so an unclosed one is a tab THEY have to close by hand.
            # After the turn has stopped (so nothing is mid-navigation on it)
            # and before the task group closes, since this awaits a subprocess.
            await self._close_browser_surface()
            # BEFORE the background tasks are cancelled: the title's own write
            # is one of them, and a name stored in the last moments of the
            # session (a `/rename` before ctrl+d) would be cancelled in flight
            # and never reach the transcript the next `--resume` reads.
            await self._flush_conversation_name()
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
            await self.jobs.dispose()
            self._wake.dispose()
            self._transcript.flush()
        finally:
            # ``finally``: host-owned resources must be released even when the
            # session's own teardown blew up part way through.
            for hook in self._dispose_hooks:
                try:
                    outcome = hook()
                    if inspect.isawaitable(outcome):
                        await outcome
                except Exception:
                    logger.warning("session dispose hook failed", exc_info=True)
