"""Fold a live session into the phone-facing projection.

One class, :class:`ProjectionFold`, fed two ways:

- **events** — the harness ``AgentEvent`` stream (``session.subscribe``),
  folded incrementally so a streaming assistant row updates in place instead
  of repainting history;
- **history** — ``session.history()`` on attach/resume, folded wholesale so a
  phone that opens mid-conversation sees the same transcript the TUI shows.

The fold owns the render semantics the TUI established and the web UI must
match exactly: one row per tool call (state glyph + one-line summary +
diff counts, details behind a tap), notices as quiet system rows, steering
receipts reconciled against the queued count. Putting those semantics here —
server-side, once — is what keeps the phone a pure renderer; the alternative
(ship raw events, fold in TypeScript) is two implementations of the TUI's
contract drifting apart.

The fold is deliberately free of asyncio: it is a plain state machine the
daemon drives from its event callback, so it is testable without a loop and
safe to call from any thread that serializes calls per session.
"""

from __future__ import annotations

import time
from typing import Any

from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    CustomMessage,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
    RetryEndEvent,
    RetryStartEvent,
    SteeringDeliveredEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
)
from local_operator.mobile.types import (
    PROJECTION_TRANSCRIPT_LIMIT,
    PendingRequest,
    SessionProjection,
    SubagentRow,
    TodoItem,
    TranscriptEntry,
)

#: How much of a tool result's text the expand payload carries. The phone's
#: expanded row is a readable window, not a log file — beyond this the right
#: surface is the terminal.
TOOL_OUTPUT_TAIL_CHARS = 8_000

#: Same bound for the args side of an expanded tool row.
TOOL_ARGS_CHARS = 4_000


def _message_text(message: AgentMessage) -> str:
    if isinstance(message, Message):
        return message.text
    if isinstance(message, CustomMessage):
        # Custom entries render their details payload's text-ish field when
        # they have one (compaction summaries, handoffs); the rest are
        # bookkeeping the transcript never showed.
        for key in ("text", "summary", "content"):
            value = message.details.get(key)
            if isinstance(value, str):
                return value
    return ""


def _summarize_args(tool_name: str, args: dict[str, Any]) -> str:
    """The one-line summary the collapsed row shows.

    Mirrors the TUI's ``_summary_from_args`` contract: the most identifying
    argument, compacted — a path, a command line, a pattern — never a dump.
    The ordering below is the TUI's priority: what a reader scans for first
    is the file or command being touched, not the options around it.
    """
    for key in ("path", "file_path", "file", "command", "pattern", "query", "url"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return _compact(value, 80)
    if args:
        first_key = next(iter(args))
        value = args[first_key]
        text = value if isinstance(value, str) else repr(value)
        return _compact(f"{first_key}={text}", 80)
    return tool_name


def _compact(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _diff_counts(details: dict[str, Any] | None) -> tuple[int, int]:
    """Green/red counts from a tool result's details, when the tool reported
    them (write/edit do). Zeroes, not guesses, everywhere else."""
    if not details:
        return 0, 0
    added = details.get("added") or details.get("lines_added") or 0
    removed = details.get("removed") or details.get("lines_removed") or 0
    try:
        return int(added), int(removed)
    except (TypeError, ValueError):
        return 0, 0


def fold_messages_to_entries(history: list[AgentMessage]) -> list[TranscriptEntry]:
    """Fold a full message history into transcript entries, UNCAPPED.

    The session-side ``ProjectionFold.fold_history`` caps to the live tail
    (the phone's realtime view is a window, not the whole log). The daemon's
    history endpoint needs the SAME render semantics over the FULL history so
    it can serve the older pages the cap dropped — this is the lazy-load
    source. Pure (no ProjectionFold state), so the daemon can call it per
    request without touching the live fold.

    Tool-result diffs ride in ``provider_payload["details"]`` (where the
    harness stores them); rehydrated messages carry that payload, so the
    write/edit rows expand to their diff exactly like the live ones.
    """
    entries: list[TranscriptEntry] = []
    # tool_call_id -> its row, local to this fold (a fresh fold re-pairs).
    tool_rows: dict[str, TranscriptEntry] = {}
    tool_args: dict[str, dict[str, Any]] = {}
    for message in history:
        if isinstance(message, CustomMessage):
            text = _message_text(message)
            if text:
                entries.append(
                    TranscriptEntry(id=message.id, kind="notice", text=_compact(text, 400))
                )
            continue
        if message.role == "user":
            entries.append(TranscriptEntry(id=message.id, kind="user", text=message.text))
        elif message.role == "assistant":
            if message.text:
                entries.append(TranscriptEntry(id=message.id, kind="assistant", text=message.text))
            for call in message.tool_calls:
                entry = TranscriptEntry(
                    id=f"{message.id}:{call.id}",
                    kind="tool",
                    tool_call_id=call.id,
                    tool_name=call.name,
                    tool_state="done",
                    summary=_summarize_args(call.name, call.arguments or {}),
                )
                entries.append(entry)
                tool_rows[call.id] = entry
                tool_args[call.id] = call.arguments or {}
        elif message.role == "tool":
            entry = tool_rows.get(message.tool_call_id or "")
            if entry is not None:
                entry.tool_state = "failed" if message.is_error else "done"
                if message.is_error:
                    entry.error = _compact(message.text, 200)
                result_details = (message.provider_payload or {}).get("details")
                entry.diff_added, entry.diff_removed = _diff_counts(result_details)
                details: dict[str, Any] = {}
                args = tool_args.get(message.tool_call_id or "", {})
                if args:
                    details["args"] = {
                        k: _compact(str(v), TOOL_ARGS_CHARS) for k, v in args.items()
                    }
                if message.text:
                    details["output"] = message.text[-TOOL_OUTPUT_TAIL_CHARS:]
                if isinstance(result_details, dict):
                    for key in ("diff", "added", "removed", "lines_added", "lines_removed"):
                        if key in result_details:
                            details[key] = result_details[key]
                entry.details = details
    return entries


class ProjectionFold:
    """Incremental fold of one session's events into a SessionProjection."""

    def __init__(self, projection: SessionProjection) -> None:
        self.projection = projection
        # tool_call_id -> transcript entry id, so start/update/end land on
        # the same row regardless of interleaving.
        self._tool_rows: dict[str, str] = {}
        self._tool_started_at: dict[str, float] = {}
        self._tool_args: dict[str, dict[str, Any]] = {}
        # The streaming assistant row, if one is open.
        self._open_message_id: str | None = None
        # The open compaction row's id, tracked explicitly the same way: a
        # reverse-scan fallback could finalize a LATER compaction's row with
        # an EARLIER end event once the tail cap starts dropping rows.
        self._open_compaction_id: str | None = None
        # Subagent roster by job id; progress updates are throttled upstream
        # (SubagentProgressEvent is never per-delta by contract).
        self._subagents: dict[str, SubagentRow] = {}
        self._subagent_started_at: dict[str, float] = {}
        # The working line's clock origin and the label's current source.
        self._activity_started_at: float | None = None

    # -- history -----------------------------------------------------------

    def fold_history(self, history: list[AgentMessage]) -> None:
        """Wholesale fold on attach: rebuild the transcript tail from the
        session's persisted history. Tool calls arrive as assistant messages
        carrying ``tool_calls`` followed by tool-role messages; we pair them
        into the same one-line rows live events would have produced."""
        entries: list[TranscriptEntry] = []
        for message in history:
            if isinstance(message, CustomMessage):
                text = _message_text(message)
                if text:
                    entries.append(
                        TranscriptEntry(id=message.id, kind="notice", text=_compact(text, 400))
                    )
                continue
            if message.role == "user":
                entries.append(TranscriptEntry(id=message.id, kind="user", text=message.text))
            elif message.role == "assistant":
                if message.text:
                    entries.append(
                        TranscriptEntry(id=message.id, kind="assistant", text=message.text)
                    )
                for call in message.tool_calls:
                    entry = TranscriptEntry(
                        id=f"{message.id}:{call.id}",
                        kind="tool",
                        tool_call_id=call.id,
                        tool_name=call.name,
                        tool_state="done",
                        summary=_summarize_args(call.name, call.arguments or {}),
                    )
                    entries.append(entry)
                    self._tool_rows[call.id] = entry.id
                    self._tool_args[call.id] = call.arguments or {}
            elif message.role == "tool":
                entry_id = self._tool_rows.get(message.tool_call_id or "")
                entry = next((e for e in entries if e.id == entry_id), None)
                if entry is not None:
                    entry.tool_state = "failed" if message.is_error else "done"
                    if message.is_error:
                        entry.error = _compact(message.text, 200)
                    entry.details = self._tool_details(
                        self._tool_args.get(message.tool_call_id or "", {}),
                        message.text,
                        None,
                    )
        # A resumed fold starts clean: no streaming row, no half-run tools.
        self._open_message_id = None
        self.projection.transcript = self._cap_tail(entries)
        # Prune the correlation maps to the surviving tail: a long history
        # pairs every historical tool call, and keeping ids whose rows were
        # cut is dead weight per rebind (and a stale hit if a call id were
        # ever reused).
        surviving = {entry.id for entry in self.projection.transcript}
        self._tool_rows = {
            call_id: entry_id
            for call_id, entry_id in self._tool_rows.items()
            if entry_id in surviving
        }
        live_call_ids = set(self._tool_rows)
        self._tool_args = {
            call_id: args for call_id, args in self._tool_args.items() if call_id in live_call_ids
        }
        self._bump()

    # -- events ------------------------------------------------------------

    def fold_event(self, event: AgentEvent) -> None:
        """Fold one live event. The dispatch is explicit if/elif rather than
        a registry so a new harness event type fails loudly here (AttributeError
        on construction import) instead of being silently dropped."""
        p = self.projection
        if isinstance(event, AgentStartEvent):
            p.streaming = True
        elif isinstance(event, AgentEndEvent):
            p.streaming = False
            p.queued_count = 0
            p.stop_reason = "aborted" if event.aborted else "completed"
            self._close_open_message()
            if event.error:
                self._append(
                    TranscriptEntry(
                        id=f"err-{time.time_ns()}", kind="notice", text=_compact(event.error, 400)
                    )
                )
        elif isinstance(event, TurnEndEvent):
            p.streaming = False
        elif isinstance(event, MessageStartEvent):
            if isinstance(event.message, Message) and event.message.role == "assistant":
                entry = TranscriptEntry(
                    id=event.message.id,
                    kind="assistant",
                    text=event.message.text,
                    final=False,
                )
                self._append(entry)
                self._open_message_id = event.message.id
            elif isinstance(event.message, Message) and event.message.role == "user":
                self.absorb_user_event(event.message)
        elif isinstance(event, MessageUpdateEvent):
            if self._open_message_id and event.message.id == self._open_message_id:
                row = self._find(self._open_message_id)
                if row is not None:
                    # Append the delta; never re-read the whole message — the
                    # delta contract is what makes 30 Hz streaming cheap.
                    row.text += event.delta
        elif isinstance(event, MessageEndEvent):
            row = self._find(event.message.id)
            if row is not None:
                row.text = _message_text(event.message)
                row.final = True
            self._open_message_id = None
        elif isinstance(event, ToolCallComposeEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            row.tool_state = "composing"
            row.summary = event.intent or f"dictating {event.tool_name}"
            row.intent = event.intent or ""
            row.details["argument_bytes"] = event.argument_bytes
        elif isinstance(event, ToolExecutionStartEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            row.tool_state = "running"
            row.summary = _summarize_args(event.tool_name, event.args)
            row.intent = event.intent or row.intent
            self._tool_started_at[event.tool_call_id] = time.monotonic()
            self._tool_args[event.tool_call_id] = event.args
        elif isinstance(event, ToolExecutionUpdateEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            text = getattr(event.partial_result, "text", "") or ""
            if text:
                row.details["partial"] = text[-TOOL_OUTPUT_TAIL_CHARS:]
        elif isinstance(event, ToolExecutionEndEvent):
            row = self._tool_row(event.tool_call_id, event.tool_name)
            result = event.result
            row.tool_state = "failed" if result.is_error else "done"
            row.elapsed_s = round(
                time.monotonic() - self._tool_started_at.pop(event.tool_call_id, time.monotonic()),
                1,
            )
            row.diff_added, row.diff_removed = _diff_counts(result.details)
            if result.is_error:
                row.error = _compact(result.text, 200)
            row.details = self._tool_details(
                self._tool_args.pop(event.tool_call_id, {}), result.text, result.details
            )
        elif isinstance(event, NoticeEvent):
            self._append(
                TranscriptEntry(
                    id=f"nt-{time.time_ns()}", kind="notice", text=_compact(event.text, 400)
                )
            )
        elif isinstance(event, SteeringDeliveredEvent):
            p.queued_count = max(0, p.queued_count - event.count)
        elif isinstance(event, SubagentStartEvent):
            self._subagents[event.job_id] = SubagentRow(
                job_id=event.job_id, label=event.label, status="running"
            )
            self._subagent_started_at[event.job_id] = time.monotonic()
        elif isinstance(event, SubagentProgressEvent):
            row = self._subagents.get(event.job_id)
            if row is not None:
                row.progress = event.progress
                row.elapsed_s = round(
                    time.monotonic() - self._subagent_started_at.get(event.job_id, time.monotonic())
                )
        elif isinstance(event, SubagentEndEvent):
            row = self._subagents.get(event.job_id)
            if row is None:
                row = SubagentRow(job_id=event.job_id, label=event.label)
                self._subagents[event.job_id] = row
            row.status = event.status  # type: ignore[assignment] — Literal matches
            row.progress = ""
            row.result_text = _compact(event.result_text or "", 200)
            row.error_text = _compact(event.error_text or "", 200)
            row.elapsed_s = round(
                time.monotonic() - self._subagent_started_at.pop(event.job_id, time.monotonic())
            )
        elif isinstance(event, CompactionStartEvent):
            entry = TranscriptEntry(
                id=f"cx-{time.time_ns()}",
                kind="compaction",
                text="compacting context…",
                final=False,
            )
            self._append(entry)
            self._open_compaction_id = entry.id
        elif isinstance(event, CompactionEndEvent):
            row = self._find(self._open_compaction_id)
            self._open_compaction_id = None
            if row is not None:
                row.final = True
                row.text = (
                    f"context compacted {event.tokens_before:,} → " f"{event.tokens_after:,} tokens"
                    if event.success
                    else "context compaction failed"
                )
        elif isinstance(event, RetryStartEvent):
            note = f"retrying ({event.attempt}): {_compact(event.error, 120)}"
            if event.fallback_model:
                note += f" — falling back to {event.fallback_model}"
            self._append(TranscriptEntry(id=f"rt-{time.time_ns()}", kind="notice", text=note))
        elif isinstance(event, RetryEndEvent):
            pass  # the retry row already reads; success is the next assistant row
        # Unknown events are dropped by design: the fold renders a SUBSET of
        # the harness taxonomy (the phone has no use for wake/loop internals),
        # and ``extra="allow"`` on AgentEvent means matching must stay
        # structural, not exhaustive-by-name.
        self._sync_subagents()
        self._derive_activity(event)
        self._bump()

    # -- user turns --------------------------------------------------------------

    def note_user_message(self, text: str, *, steer: bool = False) -> None:
        """Append the user's own message to the transcript. Called by the
        handle's prompt/steer path, because the harness only emits
        MessageStartEvent for ASSISTANT messages — a live user prompt never
        reaches the fold as an event, so without this the phone showed the
        agent's reply with no sign of what the human asked (and, for a
        phone-sent prompt, no echo of the tap at all)."""
        self._append(
            TranscriptEntry(
                id=f"user-{time.time_ns()}",
                kind="steer" if steer else "user",
                text=text,
                final=True,
            )
        )
        self._bump()

    def absorb_user_event(self, message: Message) -> bool:
        """Fold a live user ``MessageStartEvent``. The session emits these for
        user turns now, so a prompt from ANY front end reaches the fold — the
        TUI→phone direction that was missing. Returns True when it added the
        row; False when the row was already there (the handle's optimistic
        ``note_user_message`` echo for a phone-sent prompt), so the same
        message never appears twice on the phone."""
        if not isinstance(message, Message) or message.role != "user":
            return False
        text = message.text
        # De-dupe the optimistic echo: same text already sitting at the tail.
        for entry in reversed(self.projection.transcript[-3:]):
            if entry.kind in ("user", "steer") and entry.text == text:
                return False
        self._append(TranscriptEntry(id=message.id, kind="user", text=text, final=True))
        return True

    # -- working line (TUI WorkingBlock's phone counterpart) -------------------

    def _set_activity(self, label: str, *, restart_clock: bool = False) -> None:
        """Update the working line's label, resetting its clock only when the
        KIND of work changed. A tool finishing restarts the wait for the next
        model call, so the clock belongs to the phase, not the turn."""
        p = self.projection
        if restart_clock or self._activity_started_at is None or p.activity != label:
            self._activity_started_at = time.monotonic()
        p.activity = label
        p.activity_started_s = round(time.monotonic() - self._activity_started_at, 1)

    def _derive_activity(self, event: AgentEvent) -> None:
        """The label the TUI's WorkingBlock would show for this event, from the
        SAME rule: a running tool's intent, else the stream phase, else
        "thinking". Empty once the turn settles."""
        p = self.projection
        if isinstance(event, AgentStartEvent):
            self._activity_started_at = time.monotonic()
            self._set_activity("thinking")
            return
        if isinstance(event, (AgentEndEvent, TurnEndEvent)):
            p.activity = ""
            p.activity_started_s = 0.0
            self._activity_started_at = None
            return
        if not p.streaming:
            return
        if isinstance(event, ToolCallComposeEvent):
            self._set_activity(event.intent or f"dictating {event.tool_name}")
        elif isinstance(event, ToolExecutionStartEvent):
            self._set_activity(event.intent or f"running {event.tool_name}")
        elif isinstance(event, ToolExecutionEndEvent):
            # Back to waiting on the model: restart the clock for the gap.
            self._set_activity("thinking", restart_clock=True)
        elif isinstance(event, MessageStartEvent):
            if isinstance(event.message, Message) and event.message.role == "assistant":
                self._set_activity("responding")
        elif isinstance(event, MessageUpdateEvent):
            if p.activity in ("thinking", ""):
                self._set_activity("responding")

    # -- todos / pending / state -------------------------------------------

    def set_todos(self, items: list[dict[str, str]]) -> None:
        """Refresh the todo list from the tool store. Called by the owner
        after every event batch: the store is the only writer, so re-reading
        it is the fold — there is no todo event to listen for."""
        self.projection.todos = [
            TodoItem(
                text=item.get("text", ""),
                status=item.get("status", "pending"),  # type: ignore[arg-type]
                reason=item.get("reason", ""),
            )
            for item in items
        ]
        self._bump()

    def set_pending(self, pending: PendingRequest | None) -> None:
        self.projection.pending = pending
        self._bump()

    def set_state(
        self,
        *,
        model_label: str | None = None,
        model_selector: str | None = None,
        effort: str | None = None,
        effort_ladder: list[str] | None = None,
        conversation_name: str | None = None,
        cwd: str | None = None,
        queued_count: int | None = None,
        streaming: bool | None = None,
    ) -> None:
        p = self.projection
        if model_label is not None:
            p.model_label = model_label
        if model_selector is not None:
            p.model_selector = model_selector
        if effort is not None:
            p.effort = effort
        if effort_ladder is not None:
            p.effort_ladder = effort_ladder
        if conversation_name is not None:
            p.conversation_name = conversation_name
        if cwd is not None:
            p.cwd = cwd
        if queued_count is not None:
            p.queued_count = queued_count
        if streaming is not None:
            p.streaming = streaming
        self._bump()

    # -- internals ----------------------------------------------------------

    def _tool_row(self, tool_call_id: str, tool_name: str) -> TranscriptEntry:
        entry_id = self._tool_rows.get(tool_call_id)
        row = self._find(entry_id) if entry_id else None
        if row is None:
            row = TranscriptEntry(
                id=f"tc-{tool_call_id}",
                kind="tool",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            self._append(row)
            self._tool_rows[tool_call_id] = row.id
        return row

    def _tool_details(
        self, args: dict[str, Any], output: str, result_details: dict[str, Any] | None
    ) -> dict[str, Any]:
        details: dict[str, Any] = {}
        if args:
            rendered = {k: _compact(str(v), TOOL_ARGS_CHARS) for k, v in args.items()}
            details["args"] = rendered
        if output:
            details["output"] = output[-TOOL_OUTPUT_TAIL_CHARS:]
        if result_details:
            # Diff payloads ride through whole — the expanded row renders the
            # coloured unified diff from them.
            for key in ("diff", "added", "removed", "lines_added", "lines_removed"):
                if key in result_details:
                    details[key] = result_details[key]
        return details

    def _append(self, entry: TranscriptEntry) -> None:
        self.projection.transcript.append(entry)
        self.projection.transcript = self._cap_tail(self.projection.transcript)

    @staticmethod
    def _cap_tail(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Trim to the render tail WITHOUT losing the opening user message.

        A bare ``[-LIMIT:]`` drops the first user turn on any session longer
        than the cap — and the opening prompt is the one row that names what
        the whole conversation is about (and, per the field report, the row
        that was always missing). Keep the transcript's first user message
        pinned at the head, then fill the rest from the tail. The web client
        still pages older history on scroll; this is about the projection
        never omitting the conversation's own opening.
        """
        if len(entries) <= PROJECTION_TRANSCRIPT_LIMIT:
            return entries
        tail = entries[-PROJECTION_TRANSCRIPT_LIMIT:]
        first_user = next((e for e in entries if e.kind == "user"), None)
        if first_user is not None and first_user not in tail:
            # One pinned row + LIMIT-1 tail rows keeps the same bounded size.
            return [first_user, *tail[:-1]]
        return tail

    def _find(self, entry_id: str | None) -> TranscriptEntry | None:
        if not entry_id:
            return None
        for entry in reversed(self.projection.transcript):
            if entry.id == entry_id:
                return entry
        return None

    def _close_open_message(self) -> None:
        if self._open_message_id:
            row = self._find(self._open_message_id)
            if row is not None:
                row.final = True
            self._open_message_id = None

    def _sync_subagents(self) -> None:
        self.projection.subagents = sorted(
            self._subagents.values(),
            key=lambda row: (row.status != "running", row.job_id),
        )

    def _bump(self) -> None:
        self.projection.version += 1
