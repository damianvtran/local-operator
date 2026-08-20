"""The full-page subagent view — a child session read as a conversation.

A subagent's child session streams the same ``AgentEvent`` taxonomy the main
session does, and the engine retains a bounded serialization of them on the
job (``AsyncJob.trajectory``, entries are ``model_dump(mode="json")`` dicts).
This module turns that list back into a TRANSCRIPT: the child's prose becomes
:class:`AssistantBlock`\\ s, its tool calls become :class:`ToolCard`\\ s, its
notices become :class:`NoticeBlock`\\ s, all inside a real
:class:`TranscriptView`. A subagent's work therefore reads exactly like the
main conversation, because it is rendered by the same widgets under the same
adaptive-spacing and shared-name-column rules.

That is the point of the redesign. The predecessor was a small centred modal
painting its own one-line rows: prose was clipped at the card's right edge
rather than wrapped, a long run stopped mid-list with nothing saying more
existed, the app behind it was blacked out, and no row on screen said how to
get out. Three of those are properties of being a *card* rather than a *page*,
so the surface is now a page.

**A mode, not an overlay.** The view replaces the transcript region of the
LIVE screen and leaves the dock — band, status line, composer — in place, with
the parts the mode made inert greyed (``Screen.subagent`` in the stylesheet).
The user has to be able to see that this is the same app in a different mode;
a modal that blacks out the parent, or a pushed screen that hides the
composer, cannot say that. It also leaves the parent's turn painting into the
main transcript underneath, which a full-screen takeover would have had to
duplicate.

**The page ACCUMULATES; it does not mirror.** The retained trajectory is a
rolling window: past ``TRAJECTORY_CAP`` the engine deletes the oldest events,
and since every streaming delta is retained, any child that writes two or
three paragraphs sits at the cap for the rest of its life. Re-folding that
window each second yields an entry list whose head keeps *losing* content, and
mirroring it would rebuild the whole page on every tick — measured at 231 ms
and a scroll reset per second over 248 rows, on precisely the long-running
subagent this surface exists to watch. So a fold may only ever ADD to what the
page knows (:func:`_supersedes`): what was captured was true when it was
captured, and the engine forgetting it does not make it false. That is also
what keeps a reader's page intact when retention sweeps the job out from under
them five minutes after it settles.

Entries arrive from another session's internals and are read defensively: a
malformed or partial dict degrades to a skipped row, never a crash. This
surface is observability, and observability must not be able to take the app
down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from rich.cells import cell_len
from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.harness.jobs import CANCELLED_BEFORE_START
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import tool_card
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.subagent_panel import (
    SPINNER_FRAMES,
    SPINNER_INTERVAL_S,
    status_glyph,
)
from local_operator.tui.widgets.tool_card import ToolCard, truncate_cells
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    NoticeKind,
    TranscriptBlock,
    TranscriptView,
    UserBlock,
)

#: Events read per fold. The engine caps the retained list at the same number
#: (``subagent.TRAJECTORY_CAP``), so this is the TUI's half of one shared
#: bound: a view handed a longer list by a future engine still paints a
#: bounded page instead of measuring an unbounded one. Read by the fold AND
#: pinned by a test, so the two halves of the contract cannot drift silently.
TRAJECTORY_MAX_EVENTS = 500

#: Rows of a delegated brief the page shows before folding the rest away. Three
#: carries the opening sentence or two — enough to say what was asked — while
#: leaving the child's own work the majority of a 60-column body. Measured
#: against the failure it replaces: a raw 8-row paste took 36% of the body at
#: 120 columns and pushed itself entirely off a nine-step page at 60.
INSTRUCTION_ROWS = 3

#: The affordance rows, in the tool ledger's own bracketed vocabulary so the
#: gesture does not have to be learned twice. The expand row is completed with
#: the number of rows being withheld: `⟨expand⟩` alone does not distinguish two
#: more lines from fifty, and that is the difference between bothering and not.
EXPAND_HINT = tool_card.EXPAND_HINT
COLLAPSE_AFFORDANCE = tool_card.COLLAPSE_HINT

#: Cells the body's scrollbar gutter occupies, named after the
#: ``scrollbar-size-vertical: 1`` the ``TranscriptView`` rule declares. The
#: page's chrome rows are drawn by this widget and the body by Textual, so the
#: gutter is the one measurement they have to agree on by hand: without it the
#: rule ran a cell past every tool-card slab and, on a scrolled page, straight
#: over the scrollbar thumb.
SCROLLBAR_GUTTER_CELLS = 1

#: What the page says when the engine's cap has already eaten the start of the
#: run. A transcript missing its first half with no row admitting it is worse
#: than a short one: the reader draws conclusions from an opening that was
#: silently deleted.
#:
#: Keyed on the list being AT the cap, which is a proxy — the engine deletes
#: only once it is over, so a child that emitted exactly 500 events is told it
#: lost an opening it still has. That is the safe direction of the error: the
#: note costs a row, and its absence costs a wrong conclusion. A precise
#: answer needs a dropped-count on the job, which the engine does not record.
TRUNCATION_NOTE = f"earlier activity dropped — the last {TRAJECTORY_MAX_EVENTS} events are kept"

#: What the page says once the ledger has swept the job (retention is five
#: minutes past settle — an ordinary dwell time on a page whose whole purpose
#: is reading the child's conclusions). The rows above it are kept: they were
#: true when they were captured, and this row says exactly that and no more.
LEDGER_GONE_NOTE = "this subagent has left the ledger — what is above was captured before it did"

#: The mode's one always-visible statement of fact. It duplicates the
#: composer's own read-only placeholder ON PURPOSE: the placeholder is only
#: drawn while the buffer is empty, so a reader who had half a prompt typed
#: when they opened the page would otherwise have nothing on screen saying why
#: their keys do nothing.
READ_ONLY_NOTE = "read-only"


def _as_dict(event: Any) -> dict[str, Any]:
    """One trajectory entry as a dict.

    Entries are normally serialized dicts already; an engine that hands over
    live event objects is tolerated by dumping them, and anything neither is
    skipped (empty dict) rather than raised on.
    """
    if isinstance(event, dict):
        return event
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            value = dump(mode="json")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _content_text(payload: Any) -> str:
    """Text of a serialized message/result dict (text blocks only).

    ``content`` is type-checked rather than truthiness-checked: a scalar there
    is not falsy, and iterating one raised ``TypeError`` out of the 1 Hz
    refresh timer — a repeating handler exception, not the skipped row this
    module promises.
    """
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _first_line(text: str) -> str:
    """First non-empty line — what a failed tool card shows as its error."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


@dataclass(frozen=True)
class SubagentEntry:
    """One row of the folded child transcript.

    A value, not a widget. ``key`` is the row's IDENTITY across folds — the
    child's message id, its ``tool_call_id``, or the event's position for a
    notice — and everything else is the row's current content. The view merges
    successive folds by key and diffs by value, so a row has to be able to
    answer both "am I the same row" and "have I changed" without consulting
    the DOM.
    """

    key: str
    kind: Literal["prompt", "text", "tool", "notice"]
    text: str = ""
    notice_kind: NoticeKind = "info"
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    intent: str | None = None
    #: ``None`` while the call is still running — the card stays live.
    outcome: Literal["success", "error", "interrupted"] | None = None
    result_text: str = ""
    details: dict[str, Any] | None = None
    #: Rows that belong ABOVE the transcript rather than in it. Exactly one
    #: exists (the truncation note); the view mounts it once, outside the
    #: diffed sequence, so its arrival at the cap appends a row instead of
    #: renumbering every entry and rebuilding the page under the reader.
    head: bool = False


def _notice(key: str, text: str, kind: NoticeKind = "info", *, head: bool = False) -> SubagentEntry:
    return SubagentEntry(key=key, kind="notice", text=text, notice_kind=kind, head=head)


def _supersedes(new: SubagentEntry, old: SubagentEntry) -> bool:
    """May ``new`` replace what the page already knows as ``old``?

    Only when it is at least as complete. The trajectory is a rolling window,
    so a later fold of the same run can hand back a message whose opening
    deltas have been evicted or a tool whose start event is gone — a record
    that is strictly WORSE than the one already on screen. Accepting it would
    make the page lose content as the child works, which is the opposite of
    what a transcript is for.
    """
    if new.kind != old.kind:
        return True  # the row changed shape; the newer reading is the one to trust
    if new.kind == "text":
        return len(new.text) >= len(old.text)
    if new.kind == "tool":
        # A settled outcome supersedes a live one; the reverse never does.
        return new.outcome is not None or old.outcome is None
    # Notices and the parent's instruction are written once and never revised.
    # (The instruction does not reach here today — `show()` owns its key — but
    # the rule is the same one, so the fallthrough is the right home for it.)
    return False


def fold_trajectory(events: Sequence[Any], *, settled: bool = False) -> list[SubagentEntry]:
    """Fold serialized child events into transcript rows.

    Mirrors the app's own event handling one for one, because the promise of
    this surface is that a subagent's work reads like the main conversation:
    assistant messages accumulate per message id (start resets, update appends
    the delta, end adopts the authoritative text), tool rows are keyed by
    ``tool_call_id``, and compaction/retry/notice/agent_end produce the same
    notice wording the live transcript produces. Turn boundaries are noise at
    this zoom and dropped.

    ``settled`` marks the job as no longer running, and is the only way to
    tell a tool that is STILL executing from one whose end event never
    arrived. The first stays live; the second is ``interrupted``, exactly as a
    resumed conversation renders a call whose result is missing.
    """
    # Creation-ordered records. Tools are addressed by call id so a late end
    # settles the row that already printed; text is addressed by message id so
    # deltas accumulate into one block rather than one block per delta.
    ordered: list[tuple[str, str]] = []
    streams: dict[str, str] = {}
    tools: dict[str, SubagentEntry] = {}
    notices: dict[str, SubagentEntry] = {}

    def remember(kind: str, key: str) -> None:
        if (kind, key) not in ordered:
            ordered.append((kind, key))

    def note(index: int, text: str, kind: NoticeKind) -> None:
        notices[f"n{index}"] = _notice(f"n{index}", text, kind)
        remember("notice", f"n{index}")

    try:
        raw_events = list(events)
    except TypeError:
        return []  # not a sequence at all: a skipped page, never a raise
    if len(raw_events) >= TRAJECTORY_MAX_EVENTS:
        remember("notice", "truncated")
        notices["truncated"] = _notice("truncated", TRUNCATION_NOTE, "note", head=True)
    # The TAIL, not the head. The engine drops the oldest events, so the last
    # N are the ones that still exist; slicing from the front would freeze the
    # page at event 500 the moment the two caps ever diverge — while the note
    # above it claims to be showing the latest.
    for index, raw in enumerate(raw_events[-TRAJECTORY_MAX_EVENTS:]):
        event = _as_dict(raw)
        etype = event.get("type")
        if etype in ("message_start", "message_update", "message_end"):
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            message_id = str(message.get("id") or f"m{index}")
            if etype == "message_start":
                streams[message_id] = ""
                remember("text", message_id)
            elif etype == "message_update":
                if message_id not in streams:
                    streams[message_id] = ""
                    remember("text", message_id)
                streams[message_id] += str(event.get("delta") or "")
            else:  # message_end adopts the authoritative text
                text = _content_text(message) or streams.get(message_id, "")
                if message_id not in streams and not text:
                    continue
                streams[message_id] = text
                remember("text", message_id)
        elif etype == "tool_execution_start":
            # One normalisation for the whole pair. The lookup below used to
            # spell a missing id `""` while the store spelled it `"None"`, so
            # relaxing the guard between them would have settled a key nothing
            # reads. An id-less call still cannot be correlated with its end —
            # its card stays unsettled by design, which is a degraded row
            # rather than a wrong one.
            call_id = str(event.get("tool_call_id") or f"t{index}")
            args = event.get("args")
            intent = event.get("intent")
            tools[call_id] = SubagentEntry(
                key=call_id,
                kind="tool",
                tool_name=str(event.get("tool_name") or "tool"),
                tool_args=args if isinstance(args, dict) else {},
                intent=intent if isinstance(intent, str) and intent else None,
            )
            remember("tool", call_id)
        elif etype == "tool_execution_end":
            call_id = str(event.get("tool_call_id") or "")
            started = tools.get(call_id)
            if started is None:
                continue  # an end without a start: nothing to settle
            result = event.get("result")
            result = result if isinstance(result, dict) else {}
            details = result.get("details")
            tools[call_id] = SubagentEntry(
                key=call_id,
                kind="tool",
                tool_name=started.tool_name,
                tool_args=started.tool_args,
                intent=started.intent,
                outcome="error" if (event.get("is_error") or result.get("is_error")) else "success",
                result_text=_content_text(result),
                details=details if isinstance(details, dict) else None,
            )
        elif etype == "notice":
            kind = str(event.get("kind") or "info")
            note(
                index,
                str(event.get("text") or ""),
                kind if kind in ("info", "note", "success", "warning", "error") else "info",
            )
        elif etype == "compaction_start":
            note(index, "compacting context…", "info")
        elif etype == "compaction_end":
            done = bool(event.get("success"))
            note(
                index,
                "context compacted" if done else "compaction failed",
                "info" if done else "warning",
            )
        elif etype == "retry_start":
            body = f"retry {event.get('attempt', 1)}: {event.get('error', '')}".strip()
            if event.get("fallback_model"):
                body += f" → falling back to {event.get('fallback_model')}"
            note(index, body, "warning")
        elif etype == "model_change":
            # The child's route moved (fallback pinned or primary recovered).
            # One line either way: the page is a trajectory, and which model
            # answered from this point on is part of what happened.
            fell = bool(event.get("is_fallback"))
            selector = f"{event.get('provider', '')}/{event.get('model_id', '')}"
            # Same verb pairing as the parent's notices (design D2).
            note(
                index,
                (f"fell back to {selector}" if fell else f"back to {selector}"),
                "warning" if fell else "info",
            )
        elif etype == "agent_end":
            # The child's own failure, in the wording `on_turn_ended` uses for
            # the parent's. Without it a failed subagent's page simply stopped,
            # and the reason lived only on the band row the reader had left.
            if event.get("error"):
                note(index, str(event.get("error")), "error")
            elif event.get("aborted"):
                note(index, "interrupted", "warning")

    rows: list[SubagentEntry] = []
    for kind, key in ordered:
        if kind == "text":
            text = strip_control_sequences(streams.get(key, "")).strip()
            if text:  # a tool-use message carries no prose — spend no row
                rows.append(SubagentEntry(key=key, kind="text", text=text))
        elif kind == "tool":
            entry = tools[key]
            if settled and entry.outcome is None:
                # The job is over and this call never reported: that is what a
                # child killed mid-turn leaves behind, and painting it as still
                # running would claim work is happening inside a dead session.
                entry = SubagentEntry(
                    key=entry.key,
                    kind="tool",
                    tool_name=entry.tool_name,
                    tool_args=entry.tool_args,
                    intent=entry.intent,
                    outcome="interrupted",
                )
            rows.append(entry)
        else:
            entry = notices[key]
            if entry.text.strip():
                rows.append(entry)
    return rows


class InstructionBlock(UserBlock):
    """The parent's delegated instruction, COLLAPSED until asked for.

    The instruction is the user turn of this conversation, so it gets the block
    the main conversation gives a user turn — same gutter rule, same verbatim
    wrapping, no second style for one idea. Pasted whole, though, it broke the
    page in both directions at once: a task-shaped brief runs to fifty lines,
    so at 120 columns eight rows of it took 36% of the body, and on a
    nine-step child it sat eighteen rows above the viewport — present,
    invisible, and hinted at only by the scrollbar. Either way the page opened
    mid-sentence, because the body opens at the tail where the newest step is.

    So the brief is clamped to :data:`INSTRUCTION_ROWS` and the rest is one
    keystroke or click away. That is omp's answer to the same problem
    (``CollapsedSyntheticMessageComponent``: "collapse them behind a compact
    summary … real user prompts stay fully rendered"), and it is the app's own
    expansion idiom — the ``⟨expand⟩``/``⟨collapse⟩`` affordance and the
    click/Enter pair the tool ledger already teaches, so nothing new has to be
    learned to reach the rest of a brief.

    Clamping happens in :meth:`_rows`, which is where the base class does its
    own wrapping and paragraph handling: the clamp therefore inherits all four
    of ``UserBlock``'s decisions, and the height pin in its ``_build`` measures
    the clamped row count for free.
    """

    can_focus = True
    BINDINGS = [
        Binding("enter", "toggle_brief", "Expand instruction", show=False),
    ]

    def __init__(self, text: str) -> None:
        #: Set before `super().__init__`, which builds immediately.
        self._expanded = False
        self._hidden_rows = 0
        super().__init__(text)
        self.add_class("instruction-block")

    def _rows(self, body: int) -> list[str]:
        """The brief's rows, clamped to a summary plus its affordance.

        The affordance row states the COST of expanding — how many rows are
        being withheld — because "⟨expand⟩" alone does not distinguish two
        more lines from fifty, and that is the whole difference between
        clicking and not bothering.
        """
        rows = super()._rows(body)
        if self._expanded:
            self._hidden_rows = 0
            return [*rows, COLLAPSE_AFFORDANCE]
        self._hidden_rows = max(0, len(rows) - INSTRUCTION_ROWS)
        if not self._hidden_rows:
            return rows
        more = f"{EXPAND_HINT} {self._hidden_rows} more line"
        return [*rows[:INSTRUCTION_ROWS], f"{more}{'' if self._hidden_rows == 1 else 's'}"]

    def _build(self) -> RenderableType:
        """The brief behind its gutter, with the affordance row as CHROME.

        The base class paints every row in one ink (``TEXT_TOKEN``), which is
        right for a prompt where every row is the author's words. Here the last
        row is the app talking - it is an offer, not part of the brief - and at
        ``fg`` it read as the brief's closing line, which is worse than omitting
        it: a reader takes ``⟨expand⟩ 7 more lines`` for something the parent
        wrote. The loop is restated rather than hooked because the base class
        resolves its token once per build, and a per-row hook there is not mine
        to add for one caller.

        The height pin is kept for the reason its own docstring gives: a block
        that authors its own rows KNOWS its height, and letting the layout
        engine measure this one is what left a hole mid-transcript.
        """
        rule_style = Style(color=theme_mod.semantic_color(self.RULE_TOKEN))
        text_style = Style(color=theme_mod.semantic_color(self.TEXT_TOKEN))
        chrome_style = Style(color=theme_mod.semantic_color("dim"))
        body = max((self.size.width or 80) - self.RULE_COLS, self.MIN_BODY)
        gutter = self.RULE + " " * (self.RULE_COLS - cell_len(self.RULE))
        rows = self._rows(body)
        self.styles.height = len(rows)
        # `_rows` appends the affordance LAST and only when there is one to
        # offer, so its index is derivable rather than tracked - one fact.
        offer = len(rows) - 1 if self._expanded or self._hidden_rows else -1
        line = Text(no_wrap=True, overflow="ellipsis")
        for index, row in enumerate(rows):
            if index:
                line.append("\n")
            line.append(gutter, style=rule_style)
            if row:
                line.append(row, style=chrome_style if index == offer else text_style)
        return line

    def action_toggle_brief(self) -> None:
        """Show the whole brief, or fold it back to its summary.

        NOT ``action_toggle``: ``DOMNode.action_toggle(attribute_name)`` is a
        framework action that flips a named bool on the node, and shadowing it
        with a no-arg method means a ``toggle('...')`` action string aimed at
        this widget raises instead of doing what Textual documents.
        """
        if self._expanded or self._hidden_rows:
            self._expanded = not self._expanded
            self._rebuild()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        # Stopped so one gesture does not also move the transcript behind it,
        # the discipline every mouse handler in this app follows.
        event.stop()
        self.action_toggle_brief()

    def _rebuild(self) -> None:
        """Repaint at the current width and re-ask the spacing question.

        The same dance :meth:`UserBlock.on_resize` does, and for the same
        reason: this block's row count IS its content, so a toggle is a height
        change and the block below it may need its gap re-decided.
        """
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build())
        finally:
            self._finalized = was_finalized
        parent = self.parent
        if isinstance(parent, TranscriptView):
            parent.refresh_gap_around(self)


def entry_block(entry: SubagentEntry) -> TranscriptBlock:
    """One folded entry as the block the main conversation would have used.

    Tool rows settle through :meth:`ToolCard.restore` rather than
    ``mark_done``: those compute a duration from the moment the card was
    mounted, which for a replay is how long ago the page painted, not how long
    the tool took. The trajectory records no durations, so the column stays
    blank — the contract ``--resume`` replay already follows.
    """
    if entry.kind == "prompt":
        return InstructionBlock(entry.text)
    if entry.kind == "text":
        block = AssistantBlock()
        block.update_text(entry.text)
        block.finalize_text()
        return block
    if entry.kind == "notice":
        return NoticeBlock(entry.text, entry.notice_kind)
    card = ToolCard("", entry.tool_name, entry.tool_args, entry.intent)
    if entry.outcome == "error":
        card.restore(
            state="error",
            result_text=entry.result_text,
            details=entry.details,
            error=_first_line(entry.result_text),
        )
    elif entry.outcome == "interrupted":
        card.restore(state="interrupted")
    elif entry.outcome == "success":
        card.restore(state="success", result_text=entry.result_text, details=entry.details)
    else:
        # No outcome yet: the child's call is STILL GOING, so the card stays
        # live. It is restored rather than left as constructed for the same
        # reason the three settled arms are — `_started` here is when this page
        # painted the row, not when the child's tool began, and a running card
        # times itself from it. Left alone the row counted up from zero and
        # reset to zero every time `_sync_body` rebuilt it under an earlier
        # entry, which is exactly the fabricated duration this function's
        # settled arms exist to avoid.
        card.restore(state="running")
    return card


class SubagentViewDismissed(Message):
    """The page's ``esc`` hint was clicked. The app owns leaving the mode.

    A message rather than a reach into ``self.app``: the mode's teardown puts
    back the composer, the focus and the screen class, and all three belong to
    the app — the same split ``UsageDismissed`` already draws.
    """


class HintButton(Static):
    """One footer hint that DOES what it says when clicked.

    The keys were inert text: a row that reads ``esc back to conversation``
    next to a mouse that cannot use it is a caption, not an affordance, and
    the page is otherwise entirely mouse-reachable (a band row opens it, a
    tool card expands). Clicking is additive — the keyboard path is untouched
    and the hint still names the KEY, because the key is what a reader will
    reach for next time.

    Hover is repainted rather than styled. The row's ink is chosen per span in
    rich ``Text`` (key in ``dim``, label in ``faint``), so a `color` rule would
    be overridden by the style the widget builds; the same reason
    :class:`ToolCard` tracks its own hover. The background step and the HAND
    POINTER come from the stylesheet, where both can win — ``pointer: pointer``
    on ``.actionable``, which Textual writes to the terminal as OSC 22.

    ``actionable`` is not fixed at construction: an arrow with nothing left to
    scroll is inert, and a lit target that does nothing is the reported
    "nothing happens when I click" bug one step earlier. See
    :meth:`set_actionable`.
    """

    def __init__(self, key: str, action: Callable[[], None] | None = None) -> None:
        super().__init__(classes="subagent-view-hint")
        self._key = key
        self._action = action
        self._label = ""
        self._lead = False
        self._hovered = False
        self._actionable = action is not None
        self.set_class(self._actionable, "actionable")

    def paint(self, label: str, *, lead: bool) -> None:
        """Set this hint's label, and whether a ``·`` seam precedes it."""
        self._label, self._lead = label, lead
        self._repaint()

    def preview(self, label: str, *, lead: bool) -> str:
        """What this hint WOULD read as, without painting it.

        The ladder measures candidate rungs before choosing one, so it needs
        the string without the side effect of putting it on screen.
        """
        return self._build(label, lead=lead, hovered=False).plain

    def rendered(self) -> str:
        """What a reader sees on this hint right now, seam included."""
        return self._text().plain

    def _text(self) -> Text:
        return self._build(self._label, lead=self._lead, hovered=self._hovered)

    def _build(self, label: str, *, lead: bool, hovered: bool) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        # Hover lifts each tone one step, so the whole hint answers the pointer
        # rather than only its key — a half-lit hint reads as a rendering
        # glitch. The lifted tones are the SAME two tokens one rung brighter,
        # so no colour is introduced and the accent is not spent.
        if hovered and self._actionable:
            dim = Style(color=theme_mod.semantic_color("fg"))
            faint = Style(color=theme_mod.semantic_color("muted"))
        row = Text(no_wrap=True, overflow="ellipsis")
        if lead:
            # The seam belongs to the ROW, not to the hint that follows it, so
            # it keeps its resting tone even while that hint is hovered — a
            # punctuation mark must not read as part of a control.
            row.append(" · ", style=Style(color=theme_mod.semantic_color("faint")))
        if self._key:
            row.append(self._key, style=dim)
        if label:
            row.append(f" {label}" if self._key else label, style=faint)
        return row

    def _repaint(self) -> None:
        # `layout=False`: hover changes ink only and the plain text is
        # invariant under it by construction, so `Static.update`'s default
        # would invalidate the row's layout on every pointer crossing for a
        # repaint of identical width.
        self.update(self._text(), layout=False)

    def on_enter(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_hovered(True)

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_hovered(False)

    def set_actionable(self, actionable: bool) -> None:
        """Arm or disarm this hint, and drop any hover it was carrying.

        Recomputed on every chrome paint rather than fixed at construction,
        because the arrows' answer changes with the scroll offset: at the tail
        of a run, or on a transcript that fits the viewport, paging is a no-op
        and the target must stop offering itself.
        """
        if actionable == self._actionable:
            return
        self._actionable = actionable
        self.set_class(actionable, "actionable")
        if not actionable and self._hovered:
            self._hovered = False
        self._repaint()

    def _set_hovered(self, hovered: bool) -> None:
        # An inert hint never lights: `read-only` is a state rather than an
        # action, and an arrow with nothing left to scroll is the same case
        # arrived at dynamically.
        if hovered == self._hovered or not self._actionable:
            return
        self._hovered = hovered
        self._repaint()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        # Stopped so one gesture does not also reach the page beneath, the
        # same discipline every other mouse handler in this app follows.
        event.stop()
        if self._action is not None and self._actionable:
            self._action()


class SubagentView(Vertical):
    """The page: a title, a rule, the child's transcript, and the way out.

    Chrome rows are ``Static``\\ s repainted from :meth:`show`; the body is a
    real :class:`TranscriptView`, which is what buys the identical spacing,
    the shared tool-name column, the reserved scrollbar gutter, and the
    keyboard scrolling the hint advertises.

    Identified by CLASS and not by id, all the way down. Closing the page
    calls ``Widget.remove()``, which only POSTS a prune — the node lives in
    its parent's list until that message is processed — so a reopen inside
    that window mounted a second widget with the same id and raised
    ``DuplicateIds`` out of a click handler. An observability surface may not
    take the app down; classes make the collision legal.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(classes="subagent-view")
        self._job_id = job_id
        self._title = Static(classes="subagent-view-title")
        self._rule = Static(classes="subagent-view-rule")
        self._body = TranscriptView(classes="subagent-view-body")
        # One widget per hint so each can be hovered and clicked. The row
        # still sheds whole hints at narrow widths; it does that by hiding
        # widgets rather than by rebuilding one string.
        # `↑` and `↓` are SEPARATE targets. One `↑↓` hint would have to guess
        # which way a click meant, and a scroll affordance that guesses is one
        # a reader stops trusting after the first wrong jump.
        self._up_hint = HintButton("↑", lambda: self._scroll_body(down=False))
        self._down_hint = HintButton("↓", lambda: self._scroll_body(down=True))
        # The shared noun is its OWN inert hint. Hung off `↓` it made one half
        # of a symmetric gesture a 1-cell target beside an 8-cell one, told
        # nobody that clicking the word paged downward, and lit only half the
        # `↑↓ scroll` token on hover — which reads as a rendering glitch.
        self._scroll_label = HintButton("")
        self._exit_hint = HintButton("esc", self._leave)
        # The note rides the KEY span, not the label span. Through the label it
        # painted `faint` — the seam ink — which made the mode's one
        # always-visible statement of fact the least legible thing on the page.
        # It still never lights: it has no action.
        self._state_hint = HintButton(READ_ONLY_NOTE)
        self._hints = Horizontal(classes="subagent-view-hints")
        #: Everything the page KNOWS, by key and in arrival order. Merged from
        #: successive folds rather than replaced by them — see the module
        #: docstring and :func:`_supersedes`.
        self._known: dict[str, SubagentEntry] = {}
        self._order: list[str] = []
        #: The fold the mounted blocks were built from, and the blocks
        #: themselves, index for index. A refresh remounts the tail that moved
        #: and nothing else.
        self._entries: list[SubagentEntry] = []
        self._blocks: list[TranscriptBlock] = []
        #: The truncation row, mounted above the diffed sequence. Held apart
        #: because it appears at the cap, i.e. mid-read on the longest runs.
        self._head_block: NoticeBlock | None = None
        #: What the body should be showing. Distinct from `_entries`, which is
        #: what it IS showing: the two differ for exactly as long as it takes
        #: the container to mount.
        self._pending: list[SubagentEntry] = []
        self._pending_head: SubagentEntry | None = None
        #: Last painted title inputs. The spinner ticks eight times a second
        #: and everything else on the row moves once a second at most.
        self._chrome_state: tuple[Any, ...] | None = None
        #: The painted chrome, held because Textual's `Static` does not hand
        #: its content back and :meth:`rendered_rows` is the assertable form
        #: of this page.
        self._title_text = Text()
        self._rule_text = Text()
        self._hint_text = Text()
        self._hint_width: int | None = None
        self._spinner_index = 0
        self._spinner_timer: Any = None
        self._running = False
        # Title facts, defaulted so the page can paint before its first
        # `show()` — mount order is Textual's, not ours.
        self._label = ""
        #: The delegated instruction, and the raw string it was derived from.
        self._prompt_raw = ""
        self._instruction = ""
        self._status = "running"
        self._queued = False
        self._elapsed = "0s"
        #: The job's settled ``result_text``, verbatim. Read for ONE fact the
        #: page cannot otherwise know: a job cancelled while still parked never
        #: ran, so its duration is parked time and the bare word ``cancelled``
        #: beside it reads as a run that burned that long.
        self._outcome = ""

    @property
    def job_id(self) -> str:
        """The task job this page is showing."""
        return self._job_id

    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        yield self._body
        with self._hints:
            yield self._up_hint
            yield self._down_hint
            yield self._scroll_label
            yield self._exit_hint
            yield self._state_hint

    def on_mount(self) -> None:
        # The app opens the page and paints it in the same synchronous call —
        # a click handler cannot await a mount — so the first `show()` lands
        # before this container's children exist. It records the state and
        # leaves the body alone; the reconcile it skipped happens here.
        self._sync_body(self._pending, self._pending_head)
        self._paint_chrome()
        # Focus lands HERE rather than at the app's open call, for the same
        # reason: `focus()` on a widget that is not yet in the focus chain is
        # a silent no-op, which left the keystrokes the hint advertises going
        # to the composer the mode had just made inert, and left the dock lit
        # as `:focus-within` with an accent chevron over a read-only field.
        self.focus_body()
        # The arrows' answer is a function of the scroll offset, and nothing
        # else repaints when a reader scrolls — on a settled job there is no
        # spinner tick at all, so an arrow disarmed at mount (before the
        # deferred `scroll_end` lands) would stay dead until the 1 Hz poll
        # happened to notice. Watching the reactive is the only signal that
        # fires exactly when the answer changes.
        self.watch(self._body, "scroll_y", self._arm_arrows, init=False)

    def on_unmount(self) -> None:
        self._stop_spinner()

    def on_resize(self) -> None:
        # The rule spans the page and the hint sheds against it, so both are
        # functions of a width only the layout knows.
        self._paint_chrome()

    # -- content -------------------------------------------------------------
    def show(
        self,
        *,
        job_id: str,
        label: str,
        status: str,
        queued: bool,
        elapsed: str,
        outcome: str = "",
        events: Sequence[Any],
        prompt: str = "",
        progress: str = "",
    ) -> None:
        """Point the page at a job's current state and reconcile the body.

        Called on open, on every Subagent* event, and on the 1 Hz job poll, so
        it has to be cheap and it has to be idempotent: the common refresh
        changes nothing and must therefore repaint nothing.
        """
        if job_id != self._job_id:
            # Another subagent is another page. Everything the old one knew is
            # discarded, and `clear_blocks` is right HERE and nowhere else in
            # this class: its `scroll_home` is what a new page wants, and what
            # a refresh of the current one must never do to a reader.
            self._job_id = job_id
            self._known = {}
            self._order = []
            self._entries = []
            self._blocks = []
            self._head_block = None
            if self._body.is_mounted:
                self._body.clear_blocks()
        self._label = strip_control_sequences(label or job_id)
        self._status = status
        self._queued = queued
        self._elapsed = elapsed
        self._outcome = strip_control_sequences(outcome or "").strip()
        self._running = status == "running" and not queued
        gone = status == "gone"

        # The prompt leads the accumulated list and is absorbed like any other
        # entry, so it participates in the diff and is mounted exactly once.
        # Keyed, not positional: it arrives before the child's first event and
        # must not be re-mounted when that event lands.
        # Stripped once per job, not once per relayed event. `AsyncJob.prompt`
        # is deliberately unbounded and a delegated instruction runs to tens of
        # KB, so re-running the control-sequence regex on every refresh spent
        # real time producing a byte-identical value the diff then discarded.
        if prompt != self._prompt_raw:
            self._prompt_raw = prompt
            self._instruction = strip_control_sequences(prompt).strip()
        if self._instruction and "__prompt__" not in self._known:
            # HEAD position, and it is safe only because `AsyncJob.prompt` is
            # assigned at REGISTRATION (harness/subagent.py) — before a runner
            # exists, so before any trajectory entry can fold. `_order` is
            # therefore always empty here. If a future engine ever recorded the
            # prompt one refresh later, inserting at 0 would renumber a mounted
            # list and `_sync_body` would rebuild the whole page under a reader
            # mid-scroll, so the fallback appends rather than doing that.
            if self._order:
                self._order.append("__prompt__")
            else:
                self._order.insert(0, "__prompt__")
        if self._instruction:
            self._known["__prompt__"] = SubagentEntry(
                key="__prompt__", kind="prompt", text=self._instruction
            )
        head: SubagentEntry | None = None
        for entry in fold_trajectory(events, settled=not self._running and not queued):
            if entry.head:
                head = entry
                continue
            known = self._known.get(entry.key)
            if known is None:
                self._order.append(entry.key)
                self._known[entry.key] = entry
            elif _supersedes(entry, known):
                self._known[entry.key] = entry
        body = [self._known[key] for key in self._order]

        tail = self._tail_entry(gone, progress)
        if tail is not None:
            body.append(tail)
        self._pending, self._pending_head = body, head
        self._sync_body(body, head)
        self._paint_chrome()
        if self._running:
            self._start_spinner()
        else:
            self._stop_spinner()

    def _tail_entry(self, gone: bool, progress: str) -> SubagentEntry | None:
        """The row that TERMINATES the page, when the page needs one.

        Three states share this slot because they answer one question — "is
        anything still coming?" — and a page that answers it in three places
        would be three vocabularies for one fact.

        A running child needs it most. Its transcript's last block is often
        settled prose (the model thinking between tools), so without a tail
        row the bottom of a live page is indistinguishable from a finished
        one — the same failure ``TranscriptView.pin_tail`` records for the
        main conversation, where the only live thing on screen ended up in the
        scrollback.
        """
        if gone:
            return _notice("__gone__", LEDGER_GONE_NOTE, "note")
        # Rows the CHILD produced, which the parent's instruction is not. With
        # the prompt counted, a settled job that recorded no activity showed
        # the instruction and then nothing — no row admitting the run left
        # nothing behind, which is the state this slot exists to name.
        activity = any(key != "__prompt__" for key in self._order)
        if not self._running and not self._queued:
            return None if activity else _notice("__empty__", self._empty_state(), "info")
        if not activity:
            return _notice("__working__", self._empty_state(), "info")
        detail = " ".join(strip_control_sequences(progress).split())
        return _notice("__working__", f"working — {detail}" if detail else "working…", "info")

    def focus_body(self) -> None:
        """Give the scrolling body focus so ↑↓ do what the hint says."""
        self._body.focus()

    def rendered_rows(self) -> list[str]:
        """The page as plain strings — title, rule, body rows, hint.

        The assertable form of what a user reads, in the same spirit as
        ``UsagePanel.render_lines_for_test``: a test that pokes at widget
        internals passes happily while the page paints nothing.
        """
        rows: list[Any] = [self._title_text, self._rule_text]
        for block in self._body.blocks():
            # AssistantBlock renders Markdown, whose renderable is not text at
            # all; it exposes its source through `text()`, and the source is
            # what a reader sees rendered.
            text = getattr(block, "text", None)
            rows.append(text() if callable(text) else getattr(block, "renderable", ""))
        rows.append(self._hint_text)
        return [_plain(row) for row in rows]

    def _sync_body(self, entries: list[SubagentEntry], head: SubagentEntry | None) -> None:
        """Bring the mounted blocks into agreement with ``entries``.

        Diffed, never rebuilt. Because the page accumulates, the sequence is
        append-mostly by construction — the tail message grows, the tail tool
        settles — so the common refresh mounts one block and touches nothing
        else. A mid-list tool settling out of order (one batch, several calls)
        remounts from there down, which is bounded and correct.
        """
        if not self._body.is_mounted:
            return  # `on_mount` replays the pending fold once the body exists
        if head is not None and self._head_block is None:
            self._head_block = NoticeBlock(head.text, head.notice_kind)
            self._body.append_block(self._head_block)
        common = 0
        for previous, current in zip(self._entries, entries):
            if previous != current:
                break
            common += 1
        if common == len(self._entries) == len(entries):
            return
        for block in reversed(self._blocks[common:]):
            self._body.remove_block(block)
        del self._blocks[common:]
        for entry in entries[common:]:
            block = entry_block(entry)
            self._body.append_block(block)
            self._blocks.append(block)
        self._entries = list(entries)

    def _empty_state(self) -> str:
        """What the page says with nothing to show, by WHY it has nothing."""
        if self._running or self._queued:
            return "waiting for the subagent's first step…"
        return "no activity was retained for this subagent"

    # -- chrome --------------------------------------------------------------
    def _paint_chrome(self) -> None:
        width = max(1, self.size.width or 0)
        # BEFORE the memo, not inside it. The arrows' answer is a function of
        # the scroll offset, which no term of the memo carries — armed inside
        # it, an arrow disarmed at mount (nothing to scroll yet) stayed dead
        # for the life of the page.
        self._arm_arrows()
        spinner = SPINNER_FRAMES[self._spinner_index] if self._running else ""
        # The tool COUNT, not the entry count: the memo has to key on what the
        # title renders, or a row replaced by one of a different kind at equal
        # length paints a stale header. Sound on its own, rather than by way
        # of the spinner happening to repaint eight times a second.
        tools = sum(1 for entry in self._entries if entry.kind == "tool")
        state = (self._label, self._status, self._queued, self._elapsed, tools, width, spinner)
        if self._chrome_state == state:
            return
        self._chrome_state = state
        self._title_text = self._title_row(width, spinner, tools)
        self._rule_text = Text(
            "─" * max(1, width - SCROLLBAR_GUTTER_CELLS),
            style=Style(color=theme_mod.semantic_color("faint")),
        )
        # Keyed on WIDTH alone: the row is a pure function of it, while the
        # memo above also carries the spinner and therefore fires eight times
        # a second — re-measuring five candidate rungs and layout-refreshing
        # four widgets for output that cannot have changed.
        if self._hint_width != width:
            self._hint_width = width
            self._hint_text = self._hint_row(width)
        # ``layout=False`` on both: the sheet fixes each at ``height: 1``
        # (``.subagent-view-title``, ``.subagent-view-rule``) and both are
        # built ``no_wrap`` to the measured width, so neither can move the
        # box. Textual's default reflows the screen, and the memo above
        # carries the spinner, so this runs 12.5 times a second for as long as
        # the child is alive. A/B in one process, 161 blocks behind the page,
        # three-second idle windows, two rounds: 4.4%/4.2% of a core with the
        # default against 3.5%/3.6% with this.
        self._title.update(self._title_text, layout=False)
        self._rule.update(self._rule_text, layout=False)

    def _title_row(self, width: int, spinner: str, tools: int) -> Text:
        """``Subagent · <label>  <glyph> <status> · <elapsed> · <n> tools``.

        The breadcrumb is dim and the LABEL carries the base ink, which
        inverts the usage card's ``Usage  <target>`` weighting on purpose:
        there the noun is the subject and the target qualifies it, here the
        noun only says which surface you are on and the label is the title of
        the page being read — and it is the same string the band row beneath
        already paints at ``fg``.

        Seams are ``faint`` and values ``dim``, one tone per role.

        Fields DROP WHOLE rather than being truncated, on the band's own
        ladder discipline. Left to a single trailing ellipsis the row cut
        inside a value and lied: `⣿ running · 23…` cannot be told from 23s or
        23m, and `⣷ runn…` spends four cells to say less than the glyph
        already said. A field that will not fit is worth less than the one
        before it, so it leaves and the rest stay whole.
        """
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))

        if self._status == "gone":
            # No glyph, no clock, no count. The ledger has swept the job, so
            # every one of those would be invented — and the fallthrough glyph
            # was ✓, the app's own mark for "this succeeded", over a run whose
            # outcome is no longer knowable.
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append("Subagent", style=dim)
            row.append(" · ", style=faint)
            row.append(truncate_cells(self._label, max(8, width - 26)), style=fg)
            row.append("  ", style=dim)
            row.append("no longer on the ledger", style=dim)
            row.truncate(width, overflow="ellipsis")
            return row

        glyph, word, token = status_glyph(self._status, queued=self._queued, spinner_glyph=spinner)
        if self._status == "cancelled" and self._outcome == CANCELLED_BEFORE_START:
            # `⊘ cancelled · 1m36s` beside rows reading `⣷ running · 7m53s`
            # presents a PARKED wait as a run: an operator concludes the child
            # burned a minute and a half of tokens before they killed it, when
            # it burned none. The duration is right — it is the age of the job
            # — so what the title owes is the sense of it, and the manager
            # stamps exactly that phrase on the row for both surfaces to spend
            # (`harness/jobs.py`). Matched rather than sniffed: `cancel` is the
            # only writer of `result_text` on a cancelled job.
            # Two rungs, not one: the phrase is 27 cells where the bare word
            # is 9, and a single-rung ladder dropped the STATE entirely at
            # widths where the bare word still fit — the page then showed a
            # glyph and a duration with no word on screen. The ladder below
            # tries every rung with the phrase first, then every rung with the
            # bare word, so the state is the last thing to go after it has
            # already been shortened once.
            word_choices = [CANCELLED_BEFORE_START, "cancelled"]
        else:
            word_choices = [word]
        glyph_style = Style(color=theme_mod.semantic_color(token))
        # Most disposable first. The glyph is never dropped: it is the one
        # field that survives a colourless frame and the only one that still
        # answers "is this running" at a width where nothing else fits.
        # Most disposable first, and the state word is shortened before it is
        # dropped, never the reverse: dropping the word inside one pass would
        # end the ladder before the shorter variant was ever tried — the exact
        # widths between the phrase and the bare word then showed a glyph and
        # a duration with no state on screen. So each pass keeps at least the
        # word field (``len(tail)`` rungs, not ``len(tail) + 1``), every word
        # variant is tried in turn, and the wordless rung comes last.
        for keep_word in (True, False):
            for word_choice in word_choices if keep_word else [word_choices[-1]]:
                tail: list[tuple[str, Style]] = []
                if tools:
                    tail.append((f" · {tools} tool{'' if tools == 1 else 's'}", dim))
                tail.append((f" · {self._elapsed}", dim))
                tail.append((f" {word_choice}", dim))
                rungs = len(tail) if keep_word else len(tail) + 1
                for dropped in range(rungs):
                    fields = tail[dropped:]
                    # The label gets whatever the fields do not want, floored
                    # at eight cells so it never vanishes entirely — it is the
                    # page's subject. The fixed chrome is the breadcrumb AND
                    # the glyph: counting only the 13 breadcrumb cells left
                    # the budget one short, so a rung whose label consumed it
                    # exactly was rejected and the ladder fell through —
                    # non-monotone in width, with the state word visible at
                    # 35 cells and gone again at 36-40 where it still fit.
                    spend = sum(cell_len(text) for text, _ in fields) + 13 + cell_len(glyph)
                    label = truncate_cells(self._label, max(8, width - spend))
                    row = Text(no_wrap=True, overflow="ellipsis")
                    row.append("Subagent", style=dim)
                    row.append(" · ", style=faint)
                    row.append(label, style=fg)
                    row.append("  ", style=dim)
                    row.append(glyph, style=glyph_style)
                    for text, style in reversed(fields):
                        row.append(text, style=style)
                    if cell_len(row.plain) <= width:
                        return row
        # Narrower than the breadcrumb itself: keep the two things that
        # identify the page, and let the label take the ellipsis.
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(truncate_cells(self._label, max(1, width - 2)), style=fg)
        row.append(f" {glyph}", style=glyph_style)
        row.truncate(width, overflow="ellipsis")
        return row

    def _scroll_body(self, *, down: bool) -> None:
        """Move the transcript one page, the way PgUp/PgDn do.

        A PAGE and not a line, and the reason is `is_near_bottom`'s two-row
        tolerance: on a LIVE page a one-line lift off the tail still counts as
        "at the bottom", so the anchor would re-acquire on the next growth and
        the click would look like it had missed. A page clears the tolerance.

        It does mean the `↑` KEY (one line, from `ScrollableContainer`'s own
        binding) and a click on the `↑` GLYPH move by different amounts. That
        is deliberate: a pointer gesture costs more than a keypress, so it buys
        more, and the arrows are the only scroll affordance a mouse has here.

        Announced to the body as a USER scroll: the click landed on this page's
        footer rather than on the transcript, so no input handler of its own
        will see it, and an unannounced page-up is indistinguishable from the
        anchor moving itself — which would leave the child's output following
        the tail while the reader is trying to read back through it.
        """
        self._body.note_user_scroll()
        if down:
            self._body.scroll_page_down()
        else:
            self._body.scroll_page_up()

    def _leave(self) -> None:
        self.post_message(SubagentViewDismissed())

    def _arm_arrows(self, *_args: Any) -> None:
        """Arm each arrow only while it has somewhere to go.

        Cheap and idempotent (``set_actionable`` returns on no change), which
        is why it can ride the chrome paint rather than needing a watcher.
        """
        offset = self._body.scroll_offset.y
        self._up_hint.set_actionable(offset > 0)
        self._down_hint.set_actionable(offset < self._body.max_scroll_y)

    def _hint_row(self, width: int) -> Text:
        """Lay out the footer hints, shedding whole ones until the row fits.

        Five rungs. ``esc`` survives every one of them: it is the only way out
        of the mode, and a page that does not say how to leave is the exact
        complaint this redesign answers. ``read-only`` sheds FIRST — it is a
        fact the composer's own placeholder also states, while the scroll
        arrows are an affordance nothing else on screen offers, and the width
        at which the row stops fitting is the width at which scrolling starts
        mattering.

        Returns the row as one ``Text`` for :meth:`rendered_rows`; the widgets
        are what a user actually reads and clicks.
        """
        # (visible hints, esc label, state label) per rung, widest first.
        arrows = (self._up_hint, self._down_hint, self._scroll_label)
        rungs: tuple[tuple[tuple[HintButton, ...], str], ...] = (
            ((*arrows, self._exit_hint, self._state_hint), "back to conversation"),
            ((*arrows, self._exit_hint), "back to conversation"),
            ((*arrows, self._exit_hint), "back"),
            ((self._exit_hint,), "back"),
            ((self._exit_hint,), ""),
        )
        visible, esc_label = rungs[-1]
        for candidate in rungs:
            if cell_len(self._hints_text(*candidate).plain) <= width:
                visible, esc_label = candidate
                break
        self._paint_hints(visible, esc_label)
        for hint in (*arrows, self._exit_hint, self._state_hint):
            hint.display = hint in visible
        return self._hints_text(visible, esc_label)

    def _paint_hints(self, visible: tuple[HintButton, ...], esc_label: str) -> None:
        """Give each visible hint its label and its seam."""
        for hint, label, lead in self._hint_plan(visible, esc_label):
            hint.paint(label, lead=lead)

    def _hint_plan(
        self, visible: tuple[HintButton, ...], esc_label: str
    ) -> list[tuple[HintButton, str, bool]]:
        # The caption carries its own leading space. It is the one hint with no
        # key in front of it, so without it the row read `↑↓scroll` while every
        # other hint — and every hint on the aside overlay — reads `key label`.
        labels = {self._scroll_label: " scroll", self._exit_hint: esc_label}
        plan: list[tuple[HintButton, str, bool]] = []
        for position, hint in enumerate(visible):
            # `↓` follows `↑` with no seam, so the pair still reads as one
            # `↑↓ scroll` token; every other hint opens with the ` · ` seam.
            # `↓` and the shared noun follow `↑` with no seam, so the three
            # still read as one `↑↓ scroll` token; every other hint opens with
            # the ` · ` seam.
            lead = position > 0 and hint not in (self._down_hint, self._scroll_label)
            plan.append((hint, labels.get(hint, ""), lead))
        return plan

    def _hints_text(self, visible: tuple[HintButton, ...], esc_label: str) -> Text:
        """A candidate row as one string, measured before anything is painted.

        Built from the PLAN rather than from the widgets' current state: the
        ladder has to know what a rung would look like before it commits to
        one, and asking the widgets would measure the rung already on screen.
        """
        row = Text(no_wrap=True, overflow="ellipsis")
        for hint, label, lead in self._hint_plan(visible, esc_label):
            row.append(hint.preview(label, lead=lead))
        return row

    # -- spinner -------------------------------------------------------------
    def _start_spinner(self) -> None:
        # Motion, not colour, is how this app says "alive", at the cadence the
        # band and the status line already use: two speeds on one screen read
        # as two different states.
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(SPINNER_INTERVAL_S, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        self._paint_chrome()


def _plain(renderable: Any) -> str:
    """A block's text, whatever kind of renderable it happens to hold."""
    plain = getattr(renderable, "plain", None)
    if isinstance(plain, str):
        return plain
    return str(renderable)
