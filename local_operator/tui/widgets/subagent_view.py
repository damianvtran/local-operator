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

import asyncio
import hashlib
import math
import reprlib
from collections import Counter
from collections.abc import Mapping, Sequence
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
from local_operator.harness.comms import (
    HUB_COMMUNICATION_CUSTOM_TYPE,
    HUB_MESSAGE_TYPE,
    extract_parent_message,
)
from local_operator.harness.jobs import CANCELLED_BEFORE_START, TRAJECTORY_SEQ_KEY
from local_operator.session.transcript import (
    CUSTOM_KIND_CUSTOM,
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    TranscriptEntry,
    TranscriptPage,
    read_transcript_page,
)
from local_operator.tui import theme as theme_mod
from local_operator.tui.animation import BLURRED_SPINNER_INTERVAL_S, animation_focused
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
    WorkingBlock,
)

#: Events read per fold. The engine caps the retained list at the same number
#: (``subagent.TRAJECTORY_CAP``), so this is the TUI's half of one shared
#: bound: a view handed a longer list by a future engine still paints a
#: bounded page instead of measuring an unbounded one. Read by the fold AND
#: pinned by a test, so the two halves of the contract cannot drift silently.
TRAJECTORY_MAX_EVENTS = 500

#: Refreshes the first reconcile may wait for the body to be LAID OUT before it
#: gives up and folds at the fallback width anyway.
#:
#: `on_mount` runs before Textual has assigned the body a region, so the width
#: every block would fold at reads 0 and `TranscriptBlock.fold_width` falls to
#: its 80-column fallback: the page builds its first rows at 80, PINS that
#: height, and re-folds a frame later, which a reader sees as the message
#: flashing narrow on opening a page mid-stream. Measured at 140x34: 2 of 20
#: opens painted the 80-column build.
#:
#: Two, because the ordering that needs the most is mount-then-fill-in-one-beat
#: (`_open_subagent_view`'s deferred fill winning its race with layout), which
#: measured exactly two refreshes before the body reported 135. It is a bound
#: rather than a loop so a body that never gains a width paints its rows at the
#: fallback instead of never painting them at all.
SYNC_LAYOUT_WAITS = 2

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
#:
#: Worded to stay one row at the 62-column viewport that made ``kept`` an
#: orphan on line 2 of the previous phrasing (design round 1, D2). The
#: em dash is the same seam the rest of the page uses; dropping "the"/"are"
#: is what buys the cells, not a different mark.
TRUNCATION_NOTE = f"earlier activity dropped — last {TRAJECTORY_MAX_EVENTS} events kept"

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
HISTORY_PAGE_ROWS = 100
HISTORY_LOADING_NOTE = "loading earlier…"
HISTORY_START_NOTE = "transcript start"
HISTORY_UNAVAILABLE_NOTE = "history unavailable"
HISTORY_ERROR_NOTE = "load failed · Home retry"


def _as_dict(event: Any) -> Mapping[str, Any]:
    """One trajectory entry as a dict.

    Entries are normally serialized dicts already; an engine that hands over
    live event objects is tolerated by dumping them, and anything neither is
    skipped (empty dict) rather than raised on.
    """
    if isinstance(event, Mapping):
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
    if not isinstance(payload, Mapping):
        return ""
    content = payload.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return ""
    parts: list[str] = []
    for block in content:
        # Transcript encoding excludes pydantic defaults, including the text
        # block's ``type`` discriminator. The presence of ``text`` is therefore
        # the durable signal; trajectory events still carry the explicit type.
        if isinstance(block, Mapping) and (
            block.get("type") == "text" or ("text" in block and "data" not in block)
        ):
            parts.append(str(block.get("text", "")))
    return "".join(parts)


#: Characters of any single value the content fingerprint takes from each end.
#: Bounds the cost of the unstamped identity path, which runs once a second
#: over a 500-event window against payloads that have no size limit (a tool
#: result, a long error, a streamed delta). Generous enough that ordinary
#: values are fingerprinted whole. See :func:`_digest`.
_DIGEST_VALUE_CHARS = 512

#: Characters taken from the END of an over-long value, in addition to its
#: head. Small because it only has to break ties the head cannot: the
#: distinguishing detail of a long agent error (the exception type, the
#: offending field) sits at the end of a shared stack trace. See :func:`_digest`.
_DIGEST_TAIL_CHARS = 64

#: Total characters fed to the hash per event, across all of its fields.
_DIGEST_TOTAL_CHARS = 4096

#: Bounds the repr of a NON-string value. ``repr`` builds its whole result
#: before anything can slice it, so a 20 KB payload nested inside ``message``
#: or ``args`` was materialized in full on every fold even though only the
#: first characters were ever hashed (review round 2, F5). ``reprlib`` stops
#: building instead of building-then-cutting, and its own elision markers keep
#: the result stable for equal inputs, which is all identity requires.
_VALUE_REPR = reprlib.Repr()
_VALUE_REPR.maxstring = _DIGEST_VALUE_CHARS
_VALUE_REPR.maxother = _DIGEST_VALUE_CHARS
_VALUE_REPR.maxlevel = 6
for _name in ("maxdict", "maxlist", "maxtuple", "maxset", "maxfrozenset", "maxarray", "maxdeque"):
    setattr(_VALUE_REPR, _name, 16)


def _digest(event: Mapping[str, Any]) -> str:
    """A short, stable fingerprint of one event's CONTENT.

    Used only for events with no :data:`TRAJECTORY_SEQ_KEY` stamp, where the
    content is the only intrinsic thing left to identify a row by. Keys are
    visited in sorted order so two equal events fingerprint equally regardless
    of insertion order.

    BOUNDED, because this runs once a second over a 500-event window against
    payloads with no size limit: fingerprinting whole events measured 133 ms
    per fold on a window of 20 KB notices. The bound is applied while BUILDING
    the hashed string — serializing and then truncating measured *worse*
    (256 ms), since the cost is the serialization, not the hashing.

    WHAT THE BOUND MUST NOT COST is identity. A head-only cut is not safe here,
    and the failure is not theoretical: two different agent errors from one
    deep stack, or one upstream 5xx differing only in request id, share
    hundreds of leading characters and differ only at the END. Keying both to
    one fingerprint made the page DROP the second error outright — notices
    never supersede, so the later event collided with the surviving key and
    was discarded, showing the reader one stale failure where two had happened
    (review round 2, F4/D5). That is content loss in the opposite direction
    from the duplicate rows this module exists to prevent, and strictly worse.

    So a value contributes its LENGTH and its TAIL as well as its head. All
    three are O(1) on a ``str`` — a slice of a string is a view, not a copy of
    the payload — so the cost the bound bought is kept (measured 0.9 ms →
    1.1 ms per fold on that same window) while events that differ anywhere in
    length or in their last :data:`_DIGEST_TAIL_CHARS` are told apart.

    Collision is REDUCED, not eliminated: any bounded fingerprint has inputs
    that alias, and two values agreeing on length, head and tail while
    differing in the middle still share a key. The residual is not a silent
    row loss the way an unqualified head cut was, because the fields that vary
    in real events (an exception line, a field path, a request id) vary at the
    end or in length. Identity for anything a current engine relays comes from
    :data:`TRAJECTORY_SEQ_KEY`, not from here; this path exists only for
    events retained across an upgrade from a release that predates the stamp.
    """
    parts: list[str] = []
    budget = _DIGEST_TOTAL_CHARS
    try:
        for key in sorted(event, key=str):
            if budget <= 0:
                break
            value = event[key]
            # ``str`` of a string IS the string, so the slices below are views
            # rather than copies. Everything else goes through the bounded
            # repr, which stops building rather than building then cutting.
            rendered = value if isinstance(value, str) else _VALUE_REPR.repr(value)
            head = min(_DIGEST_VALUE_CHARS, budget)
            chunk = f"{key}={len(rendered)}:{rendered[:head]}"
            if len(rendered) > head:
                # Only when the head was actually cut: an untruncated value is
                # already fully represented, and appending its own tail again
                # would make the digest depend on how the budget happened to
                # fall rather than on the content.
                chunk += f"~{rendered[-_DIGEST_TAIL_CHARS:]}"
            budget -= len(chunk)
            parts.append(chunk)
        payload = "\x1f".join(parts)
    except Exception:  # pragma: no cover - exotic producer data only
        payload = repr(event)[:_DIGEST_TOTAL_CHARS]
    return hashlib.blake2s(payload.encode("utf-8", "replace"), digest_size=8).hexdigest()


class _Anchors:
    """Eviction-proof identities for the rows that have no id of their own.

    THE CONSTRAINT, and the reason this exists: ``AsyncJob.trajectory`` evicts
    from the FRONT (``harness/subagent._make_relay``), so an event's offset in
    the folded window drops by one on every later append. The view accumulates
    rows by key and never removes them (:meth:`SubagentView.show`), so any key
    derived from that offset re-spells itself as the child works and mounts a
    NEW identical row each refresh — one error notice became a dozen stacked
    copies of itself. An identity here must come from something intrinsic to
    the event, never from where it currently sits.

    Preferred source is the writer's monotonic stamp, assigned at append time
    and the one property eviction cannot touch. Absent it — events retained by
    an older release, or a hand-built fixture — the content fingerprint stands
    in, qualified by how many identical events precede it so two genuinely
    distinct notices with the same wording still occupy two rows. That ordinal
    counts over CONTENT, not over offsets, so appending cannot renumber it.

    Resolved LAZILY, per event, because the fold runs once a second over a
    500-event window and the overwhelming majority of those events carry a
    message id or a ``tool_call_id`` and never ask for an anchor at all.
    Fingerprinting the whole window up front measured ~11 ms per fold on a full
    window of id-less events, spent almost entirely on rows that then discarded
    it; the ordinal still has to be counted in window order, which is why this
    is a small stateful object rather than a plain function.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def of(self, event: Mapping[str, Any]) -> str:
        stamp = event.get(TRAJECTORY_SEQ_KEY)
        # ``bool`` is an ``int`` in Python and would alias seq 0 and 1; a
        # producer that ever writes one is malformed, not authoritative.
        if isinstance(stamp, int) and not isinstance(stamp, bool):
            return f"s{stamp}"
        # CEILING of the unstamped path, and it is deliberate: the ordinal
        # counts occurrences within the CURRENT window, so the number of rows
        # one wording can ever occupy is the most that were ever resident at
        # once, not the number of times it truly happened. Repeats spread far
        # enough apart that no two co-exist all resolve to ordinal 0 and fold
        # into the row the page already holds — five such failures render as
        # one row (design round 1, D2).
        #
        # Left as-is because the alternative is worse: a window-independent
        # counter would have to persist across folds, and any counter that
        # rises on re-reading the SAME event reintroduces exactly the
        # duplicate-row defect this module was fixed for. Under-reporting a
        # legacy trajectory is the safe direction; over-reporting is the bug.
        # Only reachable for events retained across an upgrade from a release
        # older than TRAJECTORY_SEQ_KEY, since every relayed event is stamped.
        fingerprint = _digest(event)
        ordinal = self._seen.get(fingerprint, 0)
        self._seen[fingerprint] = ordinal + 1
        return f"d{fingerprint}.{ordinal}"


def _first_line(text: str) -> str:
    """First non-empty line — what a failed tool card shows as its error."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _duration(value: Any) -> float | None:
    """A trustworthy elapsed time, or ``None`` for malformed producer data.

    JSON accepts numbers that Python also treats as booleans, while in-memory
    trajectories can carry NaN or infinities that JSON would reject. None of
    those values, nor a negative interval, describes elapsed wall time; letting
    one reach ToolCard can print nonsense or fail while formatting a replay.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    duration = float(value)
    return duration if math.isfinite(duration) and duration >= 0 else None


@dataclass(frozen=True)
class SubagentEntry:
    """One row of the folded child transcript.

    A value, not a widget. ``key`` is the row's IDENTITY across folds — the
    child's message id, its ``tool_call_id``, or an eviction-proof anchor
    (:class:`_Anchors`) for a notice and for anything the child sent
    without an id — and everything else is the row's current content. The view
    merges successive folds by key and diffs by value, so a row has to be able
    to answer both "am I the same row" and "have I changed" without consulting
    the DOM.

    The key may NOT be derived from the event's position in the retained
    window. That window slides as the engine evicts from its front, so a
    positional key renames a surviving row on every refresh, and because the
    view only ever adds rows, each new spelling mounted another copy of the
    same notice.
    """

    key: str
    kind: Literal["prompt", "user", "text", "tool", "notice", "parent_message", "subagent_message"]
    text: str = ""
    notice_kind: NoticeKind = "info"
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    intent: str | None = None
    #: ``None`` while the call is still running — the card stays live.
    outcome: Literal["success", "error", "interrupted"] | None = None
    result_text: str = ""
    details: dict[str, Any] | None = None
    duration_s: float | None = None
    #: Rows that belong ABOVE the transcript rather than in it. Exactly one
    #: exists (the truncation note); the view mounts it once, outside the
    #: diffed sequence, so its arrival at the cap appends a row instead of
    #: renumbering every entry and rebuilding the page under the reader.
    head: bool = False
    #: Is this row DONE GROWING? A text row that has had its ``message_end``,
    #: or a durable history row, which is complete by definition.
    #:
    #: Distinct from the job being settled, and that distinction is the point:
    #: a running child's transcript is mostly FINISHED messages — the prose it
    #: writes between tool calls — and deriving completeness from the job
    #: status alone left every one of them unfinalized for the whole run. They
    #: then rendered with the streaming splice's concatenated fold, which
    #: cannot produce the blank row between two paragraphs, and because a
    #: block PINS its height to the rows it authored they were pinned short
    #: and grew by one row each when the job finally settled (measured 5 -> 6
    #: and 2 -> 3 on a two-message child) — a reflow of exactly the kind this
    #: page's in-place reconciliation exists to remove. The stream carries the
    #: fact; it was only being discarded at the fold.
    complete: bool = False


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
        if old.complete and not new.complete:
            # COMPLETENESS IS AS LOSABLE AS TEXT. `message_end` is an event
            # like any other, so the rolling window evicts it too: a later
            # fold of the same message then reports it as still streaming.
            # Accepting that would un-finalize a committed block, and
            # `update_entry_block` answers a finalized block's text change
            # with a REMOUNT — so a row that had settled correctly would be
            # torn down and rebuilt mid-read. Same rule as the text length
            # below, applied to the other thing a fold can lose.
            return False
        return len(new.text) >= len(old.text)
    if new.kind == "tool":
        # A settled outcome supersedes a live one; the reverse never does.
        return new.outcome is not None or old.outcome is None
    # Notices and the parent's instruction are written once and never revised.
    # (The instruction does not reach here today — `show()` owns its key — but
    # the rule is the same one, so the fallthrough is the right home for it.)
    return False


#: Row label per to-child communication kind, shared by the journaled-fact row
#: and the extracted-envelope fallback so the same communication never reads as
#: two different actions depending on which arm rendered it. An unknown kind
#: falls back to a bare "Parent", which is honest rather than guessing.
_PARENT_LABELS = {
    "ask": "Parent · asked",
    "steer": "Parent · redirected",
    "note": "Parent",
}


def fold_transcript_entries(
    entries: Sequence[TranscriptEntry],
    *,
    tool_results: dict[str, tuple[str, bool, dict[str, Any] | None, float | None]] | None = None,
) -> list[SubagentEntry]:
    """Fold durable transcript rows into the viewer's stable row model.

    Durable rows are canonical for completed history; trajectory remains the
    live overlay. Both use message/tool IDs, so the view can reconcile pages,
    compaction replacement and the rolling trajectory window without a second
    event log or positional guesses.
    """
    folded: list[SubagentEntry] = []
    # Durable page boundaries are arbitrary. A result may arrive in the newer
    # page before its call is loaded from the older one, so retain outcomes
    # across folds instead of treating every page as a complete conversation.
    results = tool_results if tool_results is not None else {}
    for entry in entries:
        payload = entry.payload
        if entry.type == ENTRY_MESSAGE and payload.get("role") == "tool":
            call_id = str(payload.get("tool_call_id") or "")
            if call_id:
                provider_payload = payload.get("provider_payload")
                metadata = provider_payload if isinstance(provider_payload, Mapping) else {}
                details = metadata.get("details")
                duration = _duration(metadata.get("duration_s"))
                # Tool result text is the model-facing payload; presentation
                # metadata travels beside it so durable replay can restore the
                # exact same ToolCard the live parent transcript painted.
                results[call_id] = (
                    _content_text(payload),
                    bool(payload.get("is_error")),
                    dict(details) if isinstance(details, Mapping) else None,
                    duration,
                )
    communication_ids: set[str] = set()
    # Body text -> how many to-child facts are available to supersede a LEGACY
    # (id-less) envelope row. A MULTISET, not a set: the legacy arm below
    # CONSUMES one count per matched row, so steering the same words twice
    # ("focus on retries" after each of two failures is a normal operator move)
    # still renders two rows. Membership-only matching collapsed N identical
    # steers into one and under-reported history the parent actually sent.
    communication_bodies: Counter[str] = Counter()
    # Host communication facts supersede their replay-visible custom message:
    # the former include replies and correlation while the latter contain XML
    # wrappers intended for the model, never for a person. The to-child bodies
    # are retained too: a hub steer persists as a plain user row, and
    # transcripts written before steers carried their fact's id can only be
    # correlated by body text.
    #
    # CONSTRAINT — a fact may be spent at most once, and only by ITS OWN row.
    # The two correlation arms must therefore not draw on one budget during the
    # ordered walk: whichever row the walk reaches first would spend the other
    # row's fact, and in a real mixed-vintage transcript the LEGACY row is the
    # older one, so it comes first and eats the id-correlated row's fact. Two
    # delivered steers then render as one. So id matches are resolved HERE, in
    # the pre-pass over the whole loaded window, and the legacy budget is built
    # from only the facts left over. That also makes the outcome independent of
    # durable page order, which is arbitrary: a fact and its row routinely land
    # on opposite sides of a page boundary.
    entry_ids = {entry.id for entry in entries}
    to_child_facts: list[tuple[str, str]] = []
    for entry in entries:
        if (
            entry.type == ENTRY_CUSTOM
            and entry.payload.get("custom_type") == HUB_COMMUNICATION_CUSTOM_TYPE
        ):
            details = entry.payload.get("details") or {}
            communication_id = str(details.get("communication_id") or "")
            if communication_id:
                communication_ids.add(communication_id)
            if details.get("direction") == "to_child":
                fact_body = strip_control_sequences(str(details.get("body") or "")).strip()
                if fact_body:
                    to_child_facts.append((communication_id, fact_body))
    for communication_id, fact_body in to_child_facts:
        # A fact whose id names a row in this window is already CLAIMED by that
        # row, which the walk suppresses on identity alone. Withholding it from
        # the legacy budget is what stops an identical id-less row consuming it.
        # A fact with no such row (its row sits on an unloaded page, or predates
        # id-carrying steers) is what the body-text arm exists to spend.
        if communication_id and communication_id in entry_ids:
            continue
        communication_bodies[fact_body] += 1
    for entry in entries:
        payload = entry.payload
        if entry.type == ENTRY_CUSTOM:
            if payload.get("custom_type") != HUB_COMMUNICATION_CUSTOM_TYPE:
                continue
            details = payload.get("details") or {}
            direction = details.get("direction")
            body = strip_control_sequences(str(details.get("body") or "")).strip()
            if not body:
                continue
            kind = str(details.get("kind") or "")
            if direction == "to_child":
                label = _PARENT_LABELS.get(kind, "Parent")
                folded.append(
                    SubagentEntry(f"comm:{entry.id}", "parent_message", text=f"{label}\n{body}")
                )
            else:
                label = "Subagent · replied" if details.get("reply_to") else "Subagent"
                # Renders as an AssistantBlock, and a recorded communication is
                # whole the moment it exists — it never streams — so it settles
                # on arrival rather than waiting for the job.
                folded.append(
                    SubagentEntry(
                        f"comm:{entry.id}",
                        "subagent_message",
                        text=f"{label}\n\n{body}",
                        complete=True,
                    )
                )
            continue
        if entry.type != ENTRY_MESSAGE:
            continue
        if payload.get("kind") == CUSTOM_KIND_CUSTOM:
            if payload.get("custom_type") != HUB_MESSAGE_TYPE or entry.id in communication_ids:
                continue
            details = payload.get("details") or {}
            body = strip_control_sequences(str(details.get("body") or "")).strip()
            if body:
                lead = "Parent · asked" if details.get("expects_reply") else "Parent"
                folded.append(SubagentEntry(entry.id, "parent_message", text=f"{lead}\n{body}"))
            continue
        role = payload.get("role")
        text = strip_control_sequences(_content_text(payload)).strip()
        if role == "user":
            parent_message = extract_parent_message(text)
            if parent_message is not None:
                # A persisted hub steer: model-facing XML around the parent's
                # own words. The human-facing fact row supersedes it — by id
                # once steers carry their fact's id, and by body text for
                # transcripts written before that (legacy ids never match).
                # When NO fact exists in the loaded window (it may sit on an
                # unloaded page), render the extracted body as the parent row
                # instead: the XML is for the model and must never reach a
                # person either way.
                body = parent_message.body
                if entry.id in communication_ids:
                    # Correlated by id: its fact renders the row and was already
                    # withheld from the legacy budget by the pre-pass, so there
                    # is nothing to consume here.
                    continue
                if communication_bodies[body]:
                    # Consume this occurrence so a SECOND identical steer is
                    # not suppressed by the first one's fact.
                    communication_bodies[body] -= 1
                    continue
                # No fact left to supersede this row. Label by the envelope's
                # own kind rather than assuming a steer: the fallback exists
                # for transcripts this code did not write, and a note or a
                # question rendered as "redirected" would misreport what the
                # parent did.
                folded.append(
                    SubagentEntry(
                        entry.id,
                        "parent_message",
                        text=f"{_PARENT_LABELS.get(parent_message.kind, 'Parent')}\n{body}",
                    )
                )
                continue
            if text:
                folded.append(SubagentEntry(entry.id, "user", text=text))
            continue
        if role == "assistant":
            if text:
                # A durable row is a message the engine already committed to
                # the transcript, so it is complete whatever the job is doing
                # now. Saying so here is what stops a paged-in history message
                # rendering with the streaming fold on a live page.
                folded.append(SubagentEntry(entry.id, "text", text=text, complete=True))
            for raw_call in payload.get("tool_calls") or ():
                if not isinstance(raw_call, Mapping):
                    continue
                call_id = str(raw_call.get("id") or raw_call.get("tool_call_id") or "")
                if not call_id:
                    continue
                result = results.get(call_id)
                folded.append(
                    SubagentEntry(
                        call_id,
                        "tool",
                        tool_name=str(raw_call.get("name") or "tool"),
                        tool_args=dict(raw_call.get("arguments") or {}),
                        outcome=("error" if result[1] else "success") if result else None,
                        result_text=result[0] if result else "",
                        details=result[2] if result else None,
                        duration_s=result[3] if result else None,
                    )
                )
            continue
        if role == "tool":
            # Outcomes were indexed before folding so result-before-call works
            # within one page too. Tool messages never render as their own row.
            continue
    return folded


def fold_trajectory(events: Sequence[Any], *, settled: bool = False) -> list[SubagentEntry]:
    """Fold serialized child events into transcript rows.

    Mirrors the app's own event handling one for one, because the promise of
    this surface is that a subagent's work reads like the main conversation:
    assistant messages accumulate per message id (start resets, update appends
    the delta, end adopts the authoritative text), tool rows are keyed by
    ``tool_call_id``, and compaction/retry/notice/agent_end produce the same
    notice wording the live transcript produces. Turn boundaries and effective
    model changes are display-state noise at this zoom and dropped.

    ``settled`` marks the job as no longer running, and is the only way to
    tell a tool that is STILL executing from one whose end event never
    arrived. The first stays live; the second is ``interrupted``, exactly as a
    resumed conversation renders a call whose result is missing.
    """
    # Creation-ordered records. Tools are addressed by call id so a late end
    # settles the row that already printed; text is addressed by message id so
    # deltas accumulate into one block rather than one block per delta.
    ordered: list[tuple[str, str]] = []
    # A child may retain 500 events and this fold runs for every relayed event
    # plus the 1 Hz live-page poll. Membership in the ordered list made each
    # refresh quadratic in distinct rows even though insertion order is already
    # represented separately; the set keeps the same first-seen contract in O(1).
    remembered: set[tuple[str, str]] = set()
    streams: dict[str, str] = {}
    #: Message ids whose ``message_end`` arrived in THIS window. The child is
    #: done writing them even while it goes on working, which is what lets the
    #: page finalize them per row instead of waiting for the job to settle.
    finished: set[str] = set()
    tools: dict[str, SubagentEntry] = {}
    notices: dict[str, SubagentEntry] = {}
    anchors = _Anchors()

    def remember(kind: str, key: str) -> None:
        identity = (kind, key)
        if identity not in remembered:
            remembered.add(identity)
            ordered.append(identity)

    def note(event: Mapping[str, Any], text: str, kind: NoticeKind) -> None:
        # Keyed by the event's own anchor, never by its position: a notice is
        # the row this whole mechanism exists for (see :class:`_Anchors`).
        key = f"n{anchors.of(event)}"
        notices[key] = _notice(key, text, kind)
        remember("notice", key)

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
    for raw in raw_events[-TRAJECTORY_MAX_EVENTS:]:
        event = _as_dict(raw)
        etype = event.get("type")
        if etype in ("message_start", "message_update", "message_end"):
            message = event.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            # ``or`` short-circuits, so the anchor is only resolved for a
            # message the child sent without an id — the uncommon case.
            # The colon is load-bearing: without it an id-less message
            # stamped at sequence 5 keys as ``ms5``, which is a spelling a
            # real ``message.id`` can also take, and the two fold onto one
            # row. A separator makes that collision unrepresentable
            # (``m:s5`` cannot equal a producer id that is just ``ms5``).
            message_id = str(message.get("id") or f"m:{anchors.of(event)}")
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
                finished.add(message_id)
                remember("text", message_id)
        elif etype == "tool_execution_start":
            # One normalisation for the whole pair. The lookup below used to
            # spell a missing id `""` while the store spelled it `"None"`, so
            # relaxing the guard between them would have settled a key nothing
            # reads. An id-less call still cannot be correlated with its end —
            # its card stays unsettled by design, which is a degraded row
            # rather than a wrong one.
            # Same separator as the message fallback above: ``t{anchor}``
            # collides with a real ``tool_call_id`` of that spelling
            # (``ts9``), ``t:{anchor}`` cannot.
            call_id = str(event.get("tool_call_id") or f"t:{anchors.of(event)}")
            args = event.get("args")
            intent = event.get("intent")
            tools[call_id] = SubagentEntry(
                key=call_id,
                kind="tool",
                tool_name=str(event.get("tool_name") or "tool"),
                tool_args=dict(args) if isinstance(args, Mapping) else {},
                intent=intent if isinstance(intent, str) and intent else None,
            )
            remember("tool", call_id)
        elif etype == "tool_execution_end":
            call_id = str(event.get("tool_call_id") or "")
            started = tools.get(call_id)
            if started is None:
                continue  # an end without a start: nothing to settle
            result = event.get("result")
            result = result if isinstance(result, Mapping) else {}
            details = result.get("details")
            event_duration = _duration(event.get("duration_s"))
            tools[call_id] = SubagentEntry(
                key=call_id,
                kind="tool",
                tool_name=started.tool_name,
                tool_args=started.tool_args,
                intent=started.intent,
                outcome="error" if (event.get("is_error") or result.get("is_error")) else "success",
                result_text=_content_text(result),
                details=dict(details) if isinstance(details, Mapping) else None,
                duration_s=(
                    event_duration
                    if event_duration is not None
                    else _duration(result.get("duration_s"))
                ),
            )
        elif etype == "notice":
            kind = str(event.get("kind") or "info")
            note(
                event,
                str(event.get("text") or ""),
                kind if kind in ("info", "note", "success", "warning", "error") else "info",
            )
        elif etype == "compaction_start":
            note(event, "compacting context…", "info")
        elif etype == "compaction_end":
            done = bool(event.get("success"))
            note(
                event,
                "context compacted" if done else "compaction failed",
                "info" if done else "warning",
            )
        elif etype == "retry_start":
            body = f"retry {event.get('attempt', 1)}: {event.get('error', '')}".strip()
            if event.get("fallback_model"):
                body += f" → falling back to {event.get('fallback_model')}"
            note(event, body, "warning")
        elif etype == "model_change":
            # Route notices narrate the edge once; this event only keeps model
            # labels and context limits truthful elsewhere in the subagent UI.
            # Rendering it would recreate the main transcript's historical
            # two-notices-for-one-transition defect.
            continue
        elif etype == "agent_end":
            # The child's own failure, in the wording `on_turn_ended` uses for
            # the parent's. Without it a failed subagent's page simply stopped,
            # and the reason lived only on the band row the reader had left.
            if event.get("error"):
                note(event, str(event.get("error")), "error")
            elif event.get("aborted"):
                note(event, "interrupted", "warning")

    rows: list[SubagentEntry] = []
    for kind, key in ordered:
        if kind == "text":
            text = strip_control_sequences(streams.get(key, "")).strip()
            if text:  # a tool-use message carries no prose — spend no row
                # `settled` finishes the LAST message too: a child killed
                # mid-turn never sends its `message_end`, and that row is done
                # growing for the only reason that matters — nothing is left
                # to send it deltas.
                rows.append(
                    SubagentEntry(
                        key=key,
                        kind="text",
                        text=text,
                        complete=key in finished or settled,
                    )
                )
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


def _mark_consecutive_notices(rows: list[SubagentEntry]) -> list[SubagentEntry]:
    """Make a genuine consecutive repeat distinguishable from a render bug.

    Two identical notices stacked with nothing between them are the data
    model telling the truth — the child really emitted the same failure
    twice — and they are also a pixel-for-pixel two-row prefix of the
    eleven-row flood this page used to paint. A reader who learned
    "stacked identical red rows means the view is broken" will discount
    the second failure. Collapsing them into one ``×N`` row would hide
    that they are two events; leaving them byte-identical hides that they
    are two events of a different kind. An ordinal on each run of
    consecutive identical notices (same kind, same wording) is the
    smallest mark that keeps both facts on screen.

    Only consecutive runs are marked. Two identical notices with a tool
    card or a sentence between them already read as two attempts, and
    numbering those would invent a sequence the page has no right to.
    The truncation note is excluded: it is chrome, not a child event.
    """
    if len(rows) < 2:
        return rows
    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(rows):
        end = start + 1
        head = rows[start]
        if head.kind == "notice" and not head.head:
            while (
                end < len(rows)
                and rows[end].kind == "notice"
                and not rows[end].head
                and rows[end].notice_kind == head.notice_kind
                and rows[end].text == head.text
            ):
                end += 1
        if end - start >= 2:
            runs.append((start, end))
        start = end
    if not runs:
        return rows
    marked = list(rows)
    for start, end in runs:
        count = end - start
        for ordinal, index in enumerate(range(start, end), start=1):
            entry = marked[index]
            marked[index] = SubagentEntry(
                key=entry.key,
                kind="notice",
                text=f"{entry.text}  ({ordinal}/{count})",
                notice_kind=entry.notice_kind,
                head=entry.head,
            )
    return marked


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


def entry_block(
    entry: SubagentEntry, *, fold_width: int = 0, settled: bool = True
) -> TranscriptBlock:
    """One folded entry as the block the main conversation would have used.

    Tool rows settle through :meth:`ToolCard.restore` rather than
    ``mark_done``: those compute a duration from the moment the card was
    mounted, which for a replay is how long ago the page painted, not how long
    the tool took. Both trajectory end events and durable tool messages carry
    the executor's measured duration, so replay restores that authoritative
    value and leaves the column blank only for legacy rows that predate it.

    ``fold_width`` is the width of the body this block is about to be mounted
    into. Every block here bakes its width into the rows it authors AND pins
    its height to the count of them, and each one is built DETACHED — no
    parent to borrow a width from — so without this it folds at 80 columns and
    re-folds one frame later, which a reader sees as the row flashing narrow.
    Zero means "not supplied" and keeps the old fallback behaviour.

    ``settled`` says whether the JOB has stopped. A text row is committed when
    it is done growing, which is ``entry.complete or settled`` — the row's own
    ``message_end`` if it had one, else the job stopping (the last message of a
    child killed mid-turn never gets one). A row that may still receive deltas
    must NOT be finalized: a finalized ``AssistantBlock`` ignores
    ``update_text``, so finalizing on construction is what forced the page to
    rebuild a streaming message instead of updating it — and a rebuild discards
    the incremental markdown cache and re-lexes the whole message. Finalizing
    per ROW rather than per PASS is what keeps a message that finished ten
    tool calls ago from rendering with the streaming splice's fold, missing
    its paragraph breaks, until the child eventually exits.
    """
    if entry.kind == "prompt":
        return InstructionBlock(entry.text)
    if entry.kind in ("user", "parent_message"):
        return UserBlock(entry.text)
    if entry.kind in ("text", "subagent_message"):
        block = AssistantBlock()
        if fold_width:
            block.set_fold_hint(fold_width)
        block.update_text(entry.text)
        if entry.complete or settled:
            block.finalize_text()
        return block
    if entry.kind == "notice":
        if entry.key == "__working__":
            activity = entry.text or "thinking"
            return WorkingBlock(activity, activity)
        return NoticeBlock(entry.text, entry.notice_kind)
    card = ToolCard("", entry.tool_name, entry.tool_args, entry.intent)
    if fold_width:
        card.set_fold_hint(fold_width)
    if entry.outcome == "error":
        card.restore(
            state="error",
            result_text=entry.result_text,
            details=entry.details,
            error=_first_line(entry.result_text),
            duration_s=entry.duration_s,
        )
    elif entry.outcome == "interrupted":
        card.restore(state="interrupted")
    elif entry.outcome == "success":
        card.restore(
            state="success",
            result_text=entry.result_text,
            details=entry.details,
            duration_s=entry.duration_s,
        )
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


#: Keys of the rows :meth:`SubagentView._tail_entry` produces. They share one
#: slot at the foot of the page — "is anything still coming?" — and the page
#: holds that slot apart from the diffed sequence because it is positioned by
#: ROLE, not by index: it is always last, so every row the child adds moves it.
TAIL_KEYS = frozenset({"__working__", "__gone__", "__empty__"})


def _split_tail(
    entries: Sequence[SubagentEntry],
) -> tuple[list[SubagentEntry], SubagentEntry | None]:
    """Separate the terminating row, if the list ends with one."""
    if entries and entries[-1].key in TAIL_KEYS:
        return list(entries[:-1]), entries[-1]
    return list(entries), None


def update_entry_block(
    block: TranscriptBlock,
    entry: SubagentEntry,
    previous: SubagentEntry,
    *,
    settled: bool = True,
) -> bool:
    """Bring an already-mounted ``block`` up to ``entry``, or refuse.

    The counterpart to :func:`entry_block`, and the reason the page can stream
    at all. Rebuilding a row is not a cheap way to update it: a remounted
    :class:`AssistantBlock` throws away the incremental markdown cache the
    whole streaming design exists to keep, so the settling final response —
    the longest message on the page — got re-lexed in full on every tick
    (measured 7.91 ms vs 0.62 ms at 8700 characters, 12.7x), and a remounted
    :class:`WorkingBlock` restarts its shimmer sweep and its clock.

    Returns True when the block now agrees with ``entry``. False means this
    change cannot be applied in place and the caller must remount the row —
    the honest answer for the cases the widgets deliberately refuse (a settled
    tool card going back to running, a notice changing kind of block, an
    instruction's collapsed text being rewritten under the reader's toggle).
    ``previous`` is what the row was showing, which is what makes "did this
    actually change" answerable without asking the DOM.
    """
    if entry.kind != previous.kind:
        return False  # a row that changed shape is a different row
    if entry.kind in ("text", "subagent_message"):
        if not isinstance(block, AssistantBlock):
            return False
        if block.is_finalized():
            # Committed and immutable: agreeing is the whole answer, and a row
            # that has somehow gained text after settling has to be rebuilt
            # because `update_text` is a no-op on a finalized block.
            return entry.text == previous.text
        # Only ever GROWS. A shorter text at the same key is the rolling window
        # having evicted the opening deltas, which `_supersedes` already
        # refuses; reaching here with one would mean silently truncating a
        # message on screen, so the row is rebuilt rather than shrunk.
        if not entry.text.startswith(previous.text):
            return False
        if entry.text != previous.text:
            block.update_text(entry.text)
        if entry.complete or settled:
            # This ROW stopped growing — its own `message_end` arrived, or the
            # job stopped and nothing is left to send it deltas. Either is a
            # change to the row even when its text did not move: a live block
            # carries the streaming splice's concatenated fold, and committing
            # re-renders the whole message once — which is what restores the
            # blank row between paragraphs that the splice cannot produce
            # mid-stream. Checked on every pass rather than only on a text
            # change, because the refresh that observes a message end or a job
            # settle usually carries no new text at all; `finalize_text` is
            # itself idempotent, so this runs once.
            block.finalize_text()
        return True
    if entry == previous:
        return True  # identical row: the common refresh, and it repaints nothing
    if entry.kind == "tool":
        if not isinstance(block, ToolCard):
            return False
        # Identity, name and arguments are fixed at the call; only the OUTCOME
        # moves, and only ever from live to settled. `restore` is the settling
        # seam for a card the page did not run itself, and it stays the seam
        # here for exactly the reason `entry_block` documents: `mark_done`
        # would compute a duration from when this page painted the row, not
        # from how long the child's tool took. The card's `_started` is already
        # None (every card this page builds is `restore`d at construction), so
        # settling it in place cannot resurrect the fabricated-duration bug —
        # the value below is the executor's own, or blank.
        if (entry.tool_name, entry.tool_args, entry.intent) != (
            previous.tool_name,
            previous.tool_args,
            previous.intent,
        ):
            return False
        if previous.outcome is not None or entry.outcome is None:
            # Already settled, or still running with nothing new to say. A
            # settled card is finalized and must not be reopened.
            return False
        if entry.outcome == "error":
            block.restore(
                state="error",
                result_text=entry.result_text,
                details=entry.details,
                error=_first_line(entry.result_text),
                duration_s=entry.duration_s,
            )
        elif entry.outcome == "interrupted":
            block.restore(state="interrupted")
        else:
            block.restore(
                state="success",
                result_text=entry.result_text,
                details=entry.details,
                duration_s=entry.duration_s,
            )
        return True
    if entry.kind == "notice":
        if entry.key == "__working__":
            if not isinstance(block, WorkingBlock):
                return False
            # The tail line is the one row that MUST survive a refresh: it
            # owns an animation and a clock, and both restart when the widget
            # does. `set_activity` is the widget's own seam for exactly this
            # and keeps the phase clock running when only the label moved.
            activity = entry.text or "thinking"
            block.set_activity(activity, activity)
            return True
        if not isinstance(block, NoticeBlock):
            return False
        block.restate(entry.text, entry.notice_kind)
        return True
    # Prompt and user rows are written once and never revised — `_supersedes`
    # says so — so a changed one is a genuinely different row. `InstructionBlock`
    # in particular holds reader state (its expand toggle) that a rewrite would
    # silently discard.
    return False


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
        """Set this hint's label, and whether a ``·`` seam precedes it.

        Relays out when the painted CELL WIDTH changes, which is the same
        distinction :meth:`set_key` already draws and for the same reason. This
        widget is ``width: auto``, so its box is sized from the content Textual
        last measured; the no-layout path in :meth:`_repaint` exists for hover,
        where the plain text is invariant by construction, and reusing it for a
        label that GREW pins the widget to its old width and clips the new text.

        The settings page changes the label itself: opening an editor turns
        ``move`` into ``move · saves``, 8 cells to 16, and the clause naming the
        rule that an arrow key commits the value was clipped off on the frame
        where that rule first applies. It self-corrected on whatever later event
        forced a layout pass, so it presented as an intermittent flicker (2 runs
        in 8) on exactly the affordance the clause exists to teach (design round
        4, D16 / UX round 4, U21).

        Width rather than string equality, because that is what the layout
        actually depends on: relaying out for a same-width label change would
        reintroduce the invalidation cost ``_repaint`` was written to avoid.
        """
        before = cell_len(self._text().plain)
        self._label, self._lead = label, lead
        if cell_len(self._text().plain) != before:
            self.update(self._text(), layout=True)
            return
        self._repaint()

    def set_key(self, key: str) -> None:
        """Replace state copy and let Textual remeasure this auto-width hint.

        Hover only changes ink, so ordinary repaints deliberately skip layout.
        History state is different: its key grows from ``read-only`` to loading,
        error, or completion copy. Reusing the no-layout path pins the widget to
        its old width and clips a semantically important label despite free row
        space.
        """
        if key == self._key:
            return
        self._key = key
        self.update(self._text(), layout=True)

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


class SubagentTranscriptView(TranscriptView):
    """Transcript body whose top-edge command belongs to its reader page."""

    BINDINGS = [
        Binding("home", "load_earlier", "Load earlier transcript", show=False, priority=True)
    ]

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        # Focused-widget handlers precede ScrollableContainer bindings. Stop
        # Home here because its native y=0 no-op emits no edge-change signal.
        if event.key == "home":
            event.stop()
            event.prevent_default()
            self.action_load_earlier()

    def action_load_earlier(self) -> None:
        # Native ScrollableContainer Home wins while this widget has focus, but
        # at y=0 it emits no scroll change. Delegate directly so the page's
        # explicit error retry remains reachable from the advertised key.
        parent = self.parent
        if isinstance(parent, SubagentView):
            parent.action_home()


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

    BINDINGS = [
        Binding("home", "home", "Load earlier transcript", show=False),
        Binding("end", "end", "Follow live tail", show=False),
    ]

    def __init__(self, job_id: str) -> None:
        super().__init__(classes="subagent-view")
        self._job_id = job_id
        self._title = Static(classes="subagent-view-title")
        self._breadcrumb = Static(classes="subagent-view-breadcrumb")
        self._rule = Static(classes="subagent-view-rule")
        # The truncation note is PAGE chrome, not a transcript row. Mounted
        # in the body it scrolled away the moment content exceeded the
        # viewport — which is exactly when it is explaining why the run
        # starts mid-sentence (issue #407). Held here, between the rule
        # and the scrolling body, it stays on screen the way the title
        # does; display is toggled when the fold first reports the cap.
        # A real NoticeBlock so the glyph, hanging wrap and kind ink match
        # every other notice on the page rather than being a second
        # rendering of the same sentence.
        self._truncation = NoticeBlock(TRUNCATION_NOTE, "note")
        self._truncation.add_class("subagent-view-truncation")
        self._truncation.display = False
        self._body = SubagentTranscriptView(classes="subagent-view-body")
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
        self._parent_hint = HintButton("p", lambda: self._navigate("parent"))
        self._peer_hint = HintButton("[ ]", lambda: self._navigate("peer"))
        self._child_hint = HintButton("c", lambda: self._navigate("child"))
        self._root_hint = HintButton("r", lambda: self._navigate("root"))
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
        #: Refreshes the first reconcile may still wait for a laid-out body.
        #: `on_mount` fires before the body has a region, so every width the
        #: page could fold at reads 0 there and `fold_width` falls to its
        #: 80-column fallback; `_sync_body` postpones instead and replays once
        #: the body has a real width. It is a BUDGET rather than a flag
        #: because one refresh is not always enough — the mount-then-fill
        #: ordering needs two — and it is bounded rather than open so a body
        #: that never gains a width degrades to the old fallback fold instead
        #: of postponing the page's only content forever.
        self._sync_layout_waits = SYNC_LAYOUT_WAITS
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
        #: Per-row paint keys for the two chrome rows the spinner does NOT
        #: touch. `_chrome_state` above carries the spinner and so misses on
        #: essentially every tick of a running child; these two decide whether
        #: the breadcrumb and the rule have anything new to say, which on a
        #: spinner tick they never do. `None` means "never painted", so the
        #: first `_paint_chrome` always writes both.
        self._breadcrumb_key: tuple[Any, ...] | None = None
        self._rule_key: tuple[Any, ...] | None = None
        self._spinner_index = 0
        self._spinner_timer: Any = None
        #: The interval the live timer was created with. Textual timers cannot
        #: be re-rated in place, so a focus change compares against this to
        #: decide whether the timer has to be replaced at all.
        self._spinner_rate: float = SPINNER_INTERVAL_S
        self._running = False
        # Title facts, defaulted so the page can paint before its first
        # `show()` — mount order is Textual's, not ours.
        self._label = ""
        #: The delegated instruction, and the raw string it was derived from.
        self._prompt_raw = ""
        self._instruction = ""
        self._launch_message_id = ""
        #: Every deterministic ``subagent-launch:<id>`` identity this lineage
        #: owns mapped to its concise delegated instruction, including attempts
        #: #314 collapsed into the current record. Reconciliation replaces each
        #: matching durable user row with its concise prompt so no collapsed
        #: attempt leaks its full preamble (review round 4 R4-1).
        self._launch_prompts: dict[str, str] = {}
        self._status = "running"
        self._queued = False
        self._elapsed = "0s"
        #: The child's ROLE and effort TIER, recorded on the job at launch
        #: (``AsyncJob.agent_role``/``effort``). Shown in the title so the page
        #: names WHAT kind of child this is and at what level, not only its
        #: label — a scout reads very differently from a full task. Empty
        #: means not recorded; the ladder omits the field rather than inventing
        #: one, and the default "task" role is treated as noise (see
        #: ``_title_row``).
        self._agent_role = ""
        self._effort = ""
        self._ancestors: list[str] = []
        #: The job's settled ``result_text``, verbatim. Read for ONE fact the
        #: page cannot otherwise know: a job cancelled while still parked never
        #: ran, so its duration is parked time and the bare word ``cancelled``
        #: beside it reads as a run that burned that long.
        self._outcome = ""
        #: Durable transcript paging is generation-scoped because a slow disk
        #: read can finish after the reader retargets this reusable page. The
        #: generation check is the boundary that prevents child A's history
        #: from being prepended into child B.
        self._history_generation = 0
        self._history_directory: str | None = "__unresolved__"
        self._history_cursor: str | None = None
        self._history_ids: set[str] = set()
        # Folding a page in isolation cannot reconcile facts whose two durable
        # forms straddle that arbitrary boundary (tool result/call and hub
        # replay/fact are both real examples). Retain the bounded-by-user-load
        # raw window and derive display rows from the whole canonical window.
        self._history_rows: list[TranscriptEntry] = []
        self._history_tool_results: dict[
            str, tuple[str, bool, dict[str, Any] | None, float | None]
        ] = {}
        self._history_entries: list[SubagentEntry] = []
        self._history_loading = False
        self._history_exhausted = False
        self._history_error = False
        self._history_unavailable = True
        self._initial_tail_pending = False
        #: EDGE-TRIGGERED page-back latch, not a level test. ``_scroll_changed``
        #: fires for every offset the body passes through — including the
        #: anchor restore ``insert_blocks`` performs after a page prepends,
        #: and every settle frame after it. A level test there
        #: (``scroll_y <= 1`` ⇒ load) mounted a page on each of those
        #: firings while the reader sat parked at the top, and a wheel still
        #: in motion crossed straight back into the top rows after the
        #: prepend displaced it — a sustained drag walked the entire
        #: transcript (the reported "chunks load one after another at the
        #: top"). Consumed by one load; re-armed only by a USER GESTURE
        #: arriving after that load landed (``_note_history_gesture``), the
        #: one signal the mount cannot synthesize — every input path (wheel,
        #: key, arrow affordance) announces itself through the body's
        #: user-scroll hook before it moves anything, and the insert's own
        #: restore scroll never passes through that hook. ``True`` here
        #: means armed: the page opens with the reader at the tail, and the
        #: FIRST arrival at the top must load.
        self._history_at_top = True
        #: One-shot: the first laid-out body may land its sticky tail on a
        #: wrap fragment (a continuation line with no glyph). Snapped once
        #: onto a row head so the first glance is a statement; later
        #: refreshes leave a reader who has already scrolled alone.
        self._landing_snap_pending = True

    @property
    def job_id(self) -> str:
        """The task job this page is showing."""
        return self._job_id

    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._breadcrumb
        yield self._rule
        yield self._truncation
        yield self._body
        with self._hints:
            yield self._parent_hint
            yield self._peer_hint
            yield self._child_hint
            yield self._root_hint
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
        self.watch(self._body, "scroll_y", self._scroll_changed, init=False)
        # The page-back latch re-arms HERE, on the gesture signal, and nowhere
        # else (see ``_history_at_top``). The body's user-scroll hook fires for
        # every input path before it moves anything; the insert's own anchor
        # restore never passes through it, which is exactly the discrimination
        # an offset-based re-arm cannot make.
        self._body.set_on_user_scroll(self._note_history_gesture)
        self._maybe_load_history(initial=True)

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
        effective_prompt: str = "",
        launch_message_id: str = "",
        launch_prompts: dict[str, str] | None = None,
        progress: str = "",
        agent_role: str = "",
        effort: str = "",
        ancestors: Sequence[str] = (),
        transcript_directory: str | None = None,
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
            self._landing_snap_pending = True
            self._truncation.display = False
            self._reset_history(transcript_directory)
            if self._body.is_mounted:
                self._body.clear_blocks()
        elif transcript_directory != self._history_directory:
            self._reset_history(transcript_directory)
        self._label = strip_control_sequences(label or job_id)
        self._status = status
        self._queued = queued
        self._elapsed = elapsed
        self._outcome = strip_control_sequences(outcome or "").strip()
        # Role and effort are launch-time facts and never change under a running
        # child, but they still ride every refresh so a page that opened before
        # the job was fully registered picks them up on the next poll.
        self._agent_role = strip_control_sequences(agent_role or "").strip()
        self._effort = strip_control_sequences(effort or "").strip()
        self._ancestors = [strip_control_sequences(item).strip() for item in ancestors if item]
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
        # ``effective_prompt`` stays in the signature because callers still pass
        # the launch-time wrapper, but the view keeps no private copy: every
        # launch-row reconciliation keys off ``_launch_prompts`` and
        # ``_launch_message_id`` alone, so storing it here only invited a future
        # reader to think reconciliation depended on it (review round 5 M1).
        self._launch_message_id = str(launch_message_id or "").strip()
        # Concise instruction for every durable launch row this lineage owns.
        # The current launch is derived from `prompt` (kept live above); the
        # collapsed predecessors arrive already-authored from the comms node.
        # Both are stripped once here rather than on each reconcile pass.
        resolved_prompts: dict[str, str] = {}
        for key, value in (launch_prompts or {}).items():
            identity = str(key or "").strip()
            concise = strip_control_sequences(str(value or "")).strip()
            if identity and concise:
                resolved_prompts[identity] = concise
        if self._launch_message_id and self._instruction:
            resolved_prompts[self._launch_message_id] = self._instruction
        self._launch_prompts = resolved_prompts
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
        body = _mark_consecutive_notices(self._chronological_entries())

        tail = self._tail_entry(gone, progress)
        if tail is not None:
            body.append(tail)
        self._pending, self._pending_head = body, head
        self._sync_body(body, head)
        self._paint_history_state()
        self._paint_chrome()
        if self._running:
            self._start_spinner()
        else:
            self._stop_spinner()

    def _reset_history(self, directory: str | None) -> None:
        self._history_generation += 1
        self._history_directory = directory
        self._history_cursor = None
        self._history_ids = set()
        self._history_rows = []
        self._history_tool_results = {}
        self._history_entries = []
        self._history_loading = False
        self._history_exhausted = False
        self._history_error = False
        self._history_unavailable = not bool(directory)
        # Re-armed on retarget (see ``_history_at_top``): a new job is a new
        # reader at the tail whose first scroll to the top owes a page.
        self._history_at_top = True
        # ``show`` runs immediately after ``screen.mount`` but before Textual
        # has mounted this child. ``on_mount`` owns the first request; later
        # retargets are already mounted and can start immediately.
        if directory:
            self.call_after_refresh(self._maybe_load_history, initial=True)

    def _history_state_text(self) -> str:
        if self._history_loading:
            return f"{HISTORY_LOADING_NOTE} · {READ_ONLY_NOTE}"
        if self._history_error:
            return f"{HISTORY_ERROR_NOTE} · {READ_ONLY_NOTE}"
        if self._history_unavailable:
            return f"{HISTORY_UNAVAILABLE_NOTE} · {READ_ONLY_NOTE}"
        if self._history_exhausted and self._history_entries:
            return f"{HISTORY_START_NOTE} · {READ_ONLY_NOTE}"
        return READ_ONLY_NOTE

    def _paint_history_state(self) -> None:
        self._state_hint.set_key(self._history_state_text())
        self._hint_width = None
        self._chrome_state = None
        if self.is_mounted:
            self._paint_chrome()

    def _note_history_gesture(self, *_args: Any, continuous: bool = False) -> None:
        """A reader moved the body: re-arm the page-back latch.

        The re-arm half of the edge trigger (``_history_at_top`` is the
        consume half, in ``_scroll_changed``). Fired from the body's
        user-scroll hook — wheel, keys, the arrow affordances — and never
        from the insert's own restore scroll, so a page mounting cannot
        re-arm the latch its own displacement would immediately consume.

        Suppressed in two cases, both of which are a gesture that cannot
        have moved the reader OFF the top rows:

        * while a page this same gesture requested is in flight — the
          notches still arriving from that wheel land inside the top rows
          before the first page has settled, and re-arming on them made one
          drag issue a second (deduped, but real) request;
        * a CONTINUOUS gesture (wheel/scrollbar) arriving while the body is
          already clamped at the top. Such a notch moves NOTHING — the
          offset is pinned at 0 — so it is not evidence the reader left and
          came back, and re-arming on it let a held wheel load a page per
          notch while the reader sat still. A discrete act (key, affordance
          click, or a caller announcing a gesture by hand) at the top IS the
          deliberate "next page please", so it re-arms.

        A sustained drag CAN load several pages, and that is correct: each
        prepend displaces the reader a page-height down, so a wheel that
        keeps running travels that distance back up and genuinely re-arrives
        at the top — one page per real arrival. What the two suppressions
        remove is the page that no travel paid for.
        """
        if self._history_loading:
            return
        if continuous and self._body.scroll_y <= 1:
            return
        self._history_at_top = True

    def _scroll_changed(self, *_args: Any) -> None:
        self._arm_arrows()
        # EDGE, not level: consume the latch only on the crossing INTO the
        # top rows. The restore after a prepend lands the reader at the top
        # of the new height by design, and that second firing is what made a
        # level trigger cascade (see ``_history_at_top``). Staying at the top
        # — the settle frames, the error repaint, a resize — fires this watch
        # again and again and must load NOTHING.
        at_top = self._body.scroll_y <= 1
        if (
            at_top
            and self._history_at_top
            # The same first-glance gate the level trigger carried: while
            # the page still owes its reader a first look at the tail, a
            # scroll that lands at the top is the OPENING layout settling,
            # not a reader asking for history (the initial load itself can
            # move the offset to the tail and back).
            and not self._initial_tail_pending
        ):
            self._history_at_top = False
            # Edge arrival may discover another page, but an error parks here
            # until a NEW Home command explicitly accepts another disk read.
            # Otherwise the reconciliation repaint retriggers this watcher and
            # turns a persistent filesystem failure into a hot retry loop.
            self._maybe_load_history()
        # No re-arm here. The offset leaving the top rows is NOT evidence of a
        # reader leaving: the prepend itself displaces the reader down by a
        # whole page, and an offset-based re-arm let a wheel still in motion
        # walk the entire transcript. Only a GESTURE re-arms
        # (``_note_history_gesture``), and a gesture that begins below the
        # top and crosses into it is the one arrival that owes a page.

    def action_home(self) -> None:
        """Reach the earliest loaded row and request one missing page.

        Repeated Home presses while the worker is active are intentionally a
        no-op: one edge crossing means one disk read, not a queue of duplicate
        requests that later prepend the same page several times.
        """
        self._body.scroll_home(animate=False)
        # ``scroll_home`` is applied on Textual's next refresh. Passing the
        # intended offset explicitly prevents the disk worker from sampling the
        # old tail first and restoring it after the prepend, which made a real
        # keyboard Home press appear to load nothing.
        self._maybe_load_history(anchor=0.0, retry=True)

    def action_end(self) -> None:
        """Return to the live tail and re-acquire sticky following."""
        self._body.scroll_end(animate=False)

    def _maybe_load_history(
        self, *, initial: bool = False, anchor: float | None = None, retry: bool = False
    ) -> None:
        if (
            not self.is_mounted
            or self._history_loading
            or self._history_exhausted
            or self._history_unavailable
            or (self._history_error and not retry)
            or not self._history_directory
        ):
            return
        self._history_loading = True
        self._history_error = False
        if initial:
            self._initial_tail_pending = True
        self._paint_history_state()
        generation = self._history_generation
        directory = self._history_directory
        assert directory is not None  # narrowed by the guard above
        cursor = self._history_cursor
        if initial:
            anchor = None
        elif anchor is None:
            anchor = self._body.scroll_y
        self._reconcile_current_body()

        async def load() -> None:
            try:
                page = await asyncio.to_thread(
                    read_transcript_page,
                    directory,
                    before_id=cursor,
                    limit=HISTORY_PAGE_ROWS,
                )
            except FileNotFoundError:
                self._finish_history_unavailable(generation)
            except Exception:  # noqa: BLE001 — an observability surface degrades
                self._finish_history_error(generation)
            else:
                self._apply_history_page(generation, page, anchor=anchor, initial=initial)

        self.run_worker(load(), group="subagent-history", exclusive=False, exit_on_error=False)

    def _finish_history_unavailable(self, generation: int) -> None:
        if generation != self._history_generation:
            return
        self._history_loading = False
        self._history_unavailable = True
        self._paint_history_state()
        self._reconcile_current_body()
        # The initial open still owes a first glance, even when there is
        # no durable page to prepend. Without this the one-shot waits
        # forever on `_initial_tail_pending` and the wrap fragment stays.
        self._settle_initial_landing()

    def _finish_history_error(self, generation: int) -> None:
        if generation != self._history_generation:
            return
        self._history_loading = False
        self._history_error = True
        self._paint_history_state()
        self._reconcile_current_body()
        self._settle_initial_landing()

    def _settle_initial_landing(self) -> None:
        """First glance after the initial history attempt, success or not.

        The trajectory-only body is not the first glance a real child
        shows: every live job has a transcript directory, and the durable
        page prepends after the one-shot used to have already fired
        (review round 1, F1). Re-arm here, scroll to the *current* tail
        immediately, then snap onto a row head so sticky-follow cannot
        bisect a wrapping notice on the next extent change.
        """
        if not self._initial_tail_pending and not self._landing_snap_pending:
            return

        def settle() -> None:
            # Immediate, not Textual's deferred ``scroll_end``: that API
            # re-measures a frame later and lands short of a body that
            # is still growing (the same reason ``TranscriptView``
            # documents ``_scroll_to_tail``). Re-arming is load-bearing
            # — the pre-history layout may already have consumed the
            # one-shot on a trajectory-only body. Clear the history
            # gate BEFORE the snap: `_snap_landing_to_row_head` refuses
            # while `_initial_tail_pending` is set, which is what kept
            # the one-shot off the pre-history body.
            self._initial_tail_pending = False
            self._landing_snap_pending = True
            self._body._scroll_to_tail()
            self._snap_landing_to_row_head()

        self.call_after_refresh(settle)

    def _apply_history_page(
        self, generation: int, page: TranscriptPage, *, anchor: float | None, initial: bool
    ) -> None:
        if generation != self._history_generation:
            return
        # Cleared here for every reader EXCEPT the latch's in-flight
        # suppression: the PREPEND path below re-raises it until the insert's
        # own settle (`_finish_history_mount`), because that suppression must
        # span the mount, not just the disk read.
        self._history_loading = False
        self._history_error = False
        rows = list(page.entries)
        if page.reconciled:
            # Replacement is a new canonical window, not an additive page.
            # Compaction may preserve every ID while rewriting payloads, so
            # filtering through the old ID set before replacement can erase the
            # entire visible history or retain stale content.
            self._history_rows = rows
            self._history_ids = {entry.id for entry in rows}
            added_rows = rows
        else:
            added_rows = [entry for entry in rows if entry.id not in self._history_ids]
            self._history_rows[0:0] = added_rows
            self._history_ids.update(entry.id for entry in added_rows)
        # Re-derive from the accumulated raw window so superseding facts work
        # in either chronological order across any page boundary. The viewer
        # already retains one display row per loaded item, so retaining its raw
        # source does not change the user-selected paging growth class.
        self._history_tool_results = {}
        self._history_entries = fold_transcript_entries(
            self._history_rows, tool_results=self._history_tool_results
        )
        self._history_cursor = rows[0].id if rows else self._history_cursor
        self._history_exhausted = not page.has_more
        self._paint_history_state()
        self._reconcile_current_body(
            anchor=anchor,
            # A reconciled page can rewrite or remove any loaded row, so the
            # prepend-only fast path would leave stale blocks mounted beside
            # the replacement even though the model is already canonical.
            prepend=not initial and not page.reconciled and bool(added_rows),
            # Only the prepend path schedules the settle callback; carrying the
            # generation lets that callback refuse to act on a page that has
            # since been retargeted away (see `_finish_history_mount`).
            generation=generation,
        )
        if initial:
            self._settle_initial_landing()

    def _chronological_entries(self) -> list[SubagentEntry]:
        """Compose durable and live rows in conversation order.

        A resumed child's durable transcript already contains the launch turn.
        New jobs carry its exact Message/TranscriptEntry ID, so replay replaces
        that row even when later turns repeat the same words. After #314
        collapses a resumed attempt into the newest record, the transcript still
        holds every earlier attempt's ``subagent-launch:<id>`` turn, so ALL of
        them — not just the current launch — are reconciled to their concise
        authored prompt from ``self._launch_prompts``; otherwise a prior
        attempt's full role/team/system preamble leaks back as a plain user row.
        Legacy records without any launch identity keep the synthetic prompt at
        the head: duplicating old wrapper text is safer than rewriting a user
        row from a paged window that cannot prove what matching rows precede it.
        """
        live = [self._known[key] for key in self._order]
        durable_keys = {entry.key for entry in self._history_entries}
        history = list(self._history_entries)
        prompt_entry = self._known.get("__prompt__")
        # The synthetic head is the fallback for the CURRENT launch only, so its
        # suppression tracks that row specifically: a paged window holding a
        # prior attempt's launch but not the current one still needs the head.
        current_matched = False
        for index, candidate in enumerate(history):
            if candidate.kind != "user":
                continue
            concise = self._launch_prompts.get(candidate.key)
            if concise is None:
                continue
            history[index] = SubagentEntry(candidate.key, "prompt", text=concise)
            if candidate.key == self._launch_message_id:
                current_matched = True
        return [
            *([prompt_entry] if prompt_entry is not None and not current_matched else []),
            *history,
            *(
                entry
                for entry in live
                if entry.key != "__prompt__" and entry.key not in durable_keys
            ),
        ]

    def _reconcile_current_body(
        self,
        *,
        anchor: float | None = None,
        prepend: bool = False,
        generation: int | None = None,
    ) -> None:
        entries = _mark_consecutive_notices(self._chronological_entries())
        tail = self._tail_entry(self._status == "gone", "")
        if tail is not None:
            entries.append(tail)
        if prepend and self._body.is_mounted:
            # The new history page is inserted above the old history but below
            # the synthetic delegation. Find the unchanged prefix and suffix so
            # the insertion uses TranscriptView's post-layout height anchor
            # instead of rebuilding from the prompt and letting Textual shift it.
            #
            # Compared over the BODY rows only. The terminating row is pinned
            # to the foot of the transcript rather than held at an index, so
            # letting it into this comparison would make an unchanged tail
            # count toward the suffix while its block sits outside the range
            # these indices address — an off-by-one that inserts a history
            # page into the middle of the page it was meant to precede.
            old_body, old_tail = _split_tail(self._entries)
            new_body, _ = _split_tail(entries)
            prefix = 0
            for previous, current in zip(old_body, new_body):
                if previous != current:
                    break
                prefix += 1
            suffix = 0
            for previous, current in zip(reversed(old_body[prefix:]), reversed(new_body[prefix:])):
                if previous != current:
                    break
                suffix += 1
            count = len(new_body) - prefix - suffix
            if count > 0 and len(old_body) - prefix == suffix:
                new_entries = new_body[prefix : prefix + count]
                # History rows are settled by definition, and they fold at the
                # same body width every other block on this page does.
                fold = self._body.scrollable_content_region.width
                new_blocks = [entry_block(entry, fold_width=fold) for entry in new_entries]
                # The truncation note is page chrome (``_truncation``), not a
                # body block, so `prefix` and the body's index agree: there
                # is no head-offset to apply. The previous offset existed
                # because the note used to occupy body index 0 while being
                # excluded from `_entries` (review round 1, N1 / round 2, M3);
                # pinning it out of the column removes that skew entirely.
                #
                # `on_settled` re-opens the page-back latch's in-flight
                # suppression (``_note_history_gesture``): until the gaps
                # settle AND the anchor restore's own scroll has landed, the
                # notches still arriving from the wheel that requested this
                # page are part of the same gesture, and re-arming on them
                # bought a second page. Holding `_history_loading` to HERE
                # (rather than clearing it in `_apply_history_page`, before
                # the insert even mounted) makes the window match
                # `_resume_paging`'s on the parent view, which is held by the
                # same callback.
                self._body.insert_blocks(
                    prefix,
                    new_blocks,
                    anchor_offset=anchor,
                    # Bound to the generation that scheduled it: a settle
                    # callback can land after the page has been RETARGETED to
                    # another job, and an unguarded clear would take down the
                    # NEW job's `_history_loading` mid-read (every sibling
                    # completion path guards the same way).
                    # `generation` is threaded down from `_apply_history_page`
                    # (None on every non-prepend caller, which never schedule
                    # this callback). Late-binding is deliberate: the settle
                    # can land after a retarget, and the guard inside
                    # `_finish_history_mount` must compare against the
                    # generation that SCHEDULED the insert, not whatever the
                    # page is showing by then.
                    on_settled=(
                        lambda: (
                            self._finish_history_mount(generation)
                            if generation is not None
                            else None
                        )
                    ),
                )
                # Re-raised for the duration of the insert: see the comment
                # above. Cleared again by `_finish_history_mount`.
                # NOTE: this MUST NOT be visible to `_history_state_text`
                # before the reconcile below has run — `show()` repaints the
                # hint from `_history_loading` and a True here paints
                # "loading earlier…" over the just-settled "transcript
                # start". `_finish_history_mount` re-paints on clear, so the
                # hint always ends at the settled text.
                self._history_loading = True
                self._blocks[prefix:prefix] = new_blocks
                self._entries[prefix:prefix] = new_entries
                self._pending = entries
                # The tail entry itself may still have moved (a page load can
                # flip `__empty__` to nothing at all), so it is reconciled
                # through the ordinary path rather than assumed unchanged.
                if _split_tail(entries)[1] != old_tail:
                    self._sync_body(entries, self._pending_head)
                return
        self._pending = entries
        self._sync_body(entries, self._pending_head)

    def _finish_history_mount(self, generation: int) -> None:
        """The prepend path's settle callback: the page is fully mounted.

        `_apply_history_page` clears `_history_loading` synchronously, which
        is correct for every reader of the flag EXCEPT two: the page-back
        latch's in-flight suppression — that one must span the insert's own
        settle (gap settlement plus the anchor restore), or a wheel still in
        motion re-arms mid-mount — and `_history_state_text`, which renders
        "loading earlier…" for as long as the flag is set. The prepend path
        therefore re-raises the flag after `_apply_history_page`'s clear, and
        this callback is where it comes down for good, at the same moment the
        parent view's `_resume_paging` gate opens.

        The repaint matters as much as the clear: the reconcile that follows
        a prepend drives `show()`, which repaints the hint from the flag and
        painted "loading earlier…" OVER the settled "transcript start" text —
        the state was correct while the visible chrome said a read was still
        in flight, so a reader (and a test asserting the rendered page) saw a
        walk that never finished. Painting here makes the settled text the
        last word regardless of repaint order.

        Generation-guarded like every other history completion path: this is
        a deferred callback, so it can land after `show()` has retargeted the
        page to a DIFFERENT job (`_reset_history` bumped the generation and
        started that job's own initial read). Clearing unguarded would take
        the new job's `_history_loading` down mid-read, leaving its hint on
        "loading earlier…" forever and its latch un-suppressed.
        """
        if generation != self._history_generation:
            return
        self._history_loading = False
        self._paint_history_state()

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
        activity = bool(self._history_entries) or any(key != "__prompt__" for key in self._order)
        if self._history_loading and not activity:
            return None
        if not self._running and not self._queued:
            return None if activity else _notice("__empty__", self._empty_state(), "info")
        detail = " ".join(strip_control_sequences(progress).split())
        # The parent transcript's aggregate progress block owns animation,
        # duration, and compact activity grammar. The child supplies only the
        # observed activity, never a parallel notice vocabulary.
        return _notice("__working__", detail or "thinking", "info")

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
        if self._truncation.display:
            rows.append(self._truncation.text())
        for block in self._body.blocks():
            # AssistantBlock renders Markdown, whose renderable is not text at
            # all; it exposes its source through `text()`, and the source is
            # what a reader sees rendered.
            text = getattr(block, "text", None)
            rows.append(text() if callable(text) else getattr(block, "renderable", ""))
        rows.append(self._hint_text)
        rows.insert(1, getattr(self._breadcrumb, "renderable", ""))
        return [_plain(row) for row in rows]

    def _replay_pending_sync(self) -> None:
        """Reconcile the fold that was postponed for want of a laid-out width.

        Runs one refresh after the deferral, by which point Textual has given
        the body its region. It reads ``_pending`` rather than closing over the
        entries it was scheduled with, so a refresh that lands in the gap wins:
        the page paints what is CURRENT at the moment it can paint correctly,
        never a snapshot that is already one tick stale.
        """
        self._sync_body(self._pending, self._pending_head)

    def _sync_body(self, entries: list[SubagentEntry], head: SubagentEntry | None) -> None:
        """Bring the mounted blocks into agreement with ``entries``.

        Reconciled IN PLACE, keyed on identity. The predecessor diffed by
        VALUE: it found the longest prefix of rows that compared equal and
        rebuilt everything after it. A growing message never compares equal to
        itself, so the row being streamed was destroyed and rebuilt on every
        tick, and the tail working line under it went with it — 240 mounts and
        239 removes over 120 streaming ticks, where one mount and no removes
        is the correct answer. What the user saw was the message flashing at
        the fallback fold width for a frame (a rebuilt block is built
        unmounted) and the working line's shimmer and clock restarting under
        it. A mid-list row settling was worse than linear: rebuilding the
        suffix beneath it made a settling batch of 16 parallel calls cost 272
        mounts.

        So the sequence is matched by KEY — the identity ``SubagentEntry``
        already promises — and a row whose content moved is UPDATED through
        :func:`update_entry_block`. Remount is reserved for what genuinely
        needs it: a new row, a row whose kind changed, a reordering, and the
        changes the widgets refuse in place.

        Everything the value diff guaranteed is preserved. The ``__prompt__``
        head keeps its position because it keeps its key; the truncation row
        is still mounted once, outside this sequence; a reader who has
        scrolled up is not moved, because an in-place update mounts nothing;
        and no row is ever dropped, because the fold that produced ``entries``
        has already been through :func:`_supersedes`.

        The FIRST pass is deferred by a frame when the body has not been laid
        out yet — see the guard below. Reconciling into a zero-width body is
        not cheaper than waiting, it is just wrong a frame earlier.
        """
        if not self._body.is_mounted:
            return  # `on_mount` replays the pending fold once the body exists
        # The width every block built below is about to be mounted at. Read
        # once per pass from the body's scrollable content region — the blocks
        # are `width: 1fr` children of it — because each is constructed
        # DETACHED and would otherwise fold at 80 columns and re-fold a frame
        # later, visibly.
        fold = self._body.scrollable_content_region.width
        if not fold and entries and self._sync_layout_waits > 0:
            # MOUNTED IS NOT LAID OUT. `on_mount` runs before Textual has
            # assigned this body a region, so every width the page could ask
            # for — the body's size, its container, the view's own — reads 0,
            # and a fold of 0 sends `fold_width` to its 80-column fallback.
            # Measured over 20 opens at 140x34: 2 of them built the page's
            # first blocks at 80 and re-folded them to 135 a frame later, and
            # the compositor painted the 80-column build, which is the width
            # flash on opening a page mid-stream.
            #
            # The screen's width is the one non-zero source here (138) and it
            # is deliberately NOT used: it is the whole terminal, 3 cells wider
            # than the body's 135, so a hint from it folds too wide and clips.
            # Waiting one frame is the only way to build at a width that is
            # RIGHT rather than merely non-zero. It costs a frame of empty
            # body, not a reflow — the rows appear once, already correct,
            # instead of appearing narrow and re-folding under the reader.
            #
            # A BUDGET, not a one-shot: the mount-then-fill ordering needs two
            # refreshes before the body has a region, and a single deferral
            # replayed into a still-zero width and folded at 80 anyway. Bounded
            # so a body that never gains a width (a page closed under the
            # deferral, a zero-height layout) falls through to the old fallback
            # behaviour and paints SOMETHING rather than postponing forever.
            self._sync_layout_waits -= 1
            self.call_after_refresh(self._replay_pending_sync)
            return
        # A row may still receive deltas while the child is working, so a text
        # block is committed once the row itself is done — `entry.complete`,
        # i.e. its `message_end` arrived — or once the job stops, which
        # finishes the last message of a child that was killed mid-turn.
        # Finalizing earlier is what made a streaming message unupdatable and
        # therefore rebuilt; finalizing per PASS instead of per ROW is what
        # left finished messages folded as if they were still streaming.
        settled = not self._running and not self._queued
        if head is not None and self._head_block is None:
            # Painted as page chrome (``_truncation``), not as a body
            # block: a body-mounted note is an ordinary row at index 0
            # and leaves the screen the moment the transcript exceeds
            # the viewport. ``_head_block`` stays the same object
            # ``rendered_rows`` and the history-prepend tests already
            # hold; it is just never mounted into the scrolling column.
            self._head_block = self._truncation
            self._truncation.restate(head.text, head.notice_kind)
            self._truncation.display = True

        # The terminating row is held OUT of the diffed sequence, for the same
        # reason the truncation row is: it is positioned by role rather than by
        # index. It is always last, so every row the child adds shifts it, and
        # a positional walk would see that shift as the tail row having changed
        # and rebuild it — destroying the one widget on the page that owns an
        # animation and a clock. Pinning it (`TranscriptView.pin_tail`, which
        # the main conversation already uses for its own working line) makes
        # later rows mount ABOVE it instead, so the widget is never disturbed.
        old_body, old_tail_entry = _split_tail(self._entries)
        new_body, new_tail_entry = _split_tail(entries)
        blocks = self._blocks
        old_tail_block = blocks[len(old_body)] if old_tail_entry is not None else None
        body_blocks = blocks[: len(old_body)]

        # Positional and keyed, not by value. Order is content here — two rows
        # swapping places is a different page — so the walk stops at the first
        # position whose identity differs or whose change the widget refuses,
        # and only from there down is anything remounted.
        common = 0
        limit = min(len(old_body), len(new_body))
        while common < limit:
            was, now = old_body[common], new_body[common]
            if was.key != now.key:
                break
            if not update_entry_block(body_blocks[common], now, was, settled=settled):
                break
            common += 1

        for block in reversed(body_blocks[common:]):
            self._body.remove_block(block)
        del body_blocks[common:]
        if body_blocks[common:] or new_body[common:]:
            # Opening a retained 500-event child can add hundreds of rows at
            # once. Mount them as one Textual batch so stylesheet/layout
            # settlement happens once; live refreshes still append their
            # ordinary one-row suffix. (A pinned tail suspends the batching —
            # see `batch_append` — which is why the tail is reconciled after
            # this, so the first fill of a page still batches.)
            with self._body.batch_append():
                for entry in new_body[common:]:
                    block = entry_block(entry, fold_width=fold, settled=settled)
                    self._body.append_block(block)
                    body_blocks.append(block)

        tail_block = old_tail_block
        if old_tail_entry is not None and new_tail_entry is not None:
            if old_tail_entry.key != new_tail_entry.key or not update_entry_block(
                old_tail_block, new_tail_entry, old_tail_entry  # type: ignore[arg-type]
            ):
                # A change of terminating STATE (running → gone, say) is a
                # different row, so it is rebuilt; a change of activity inside
                # one state was already applied in place above.
                self._body.remove_block(old_tail_block)  # type: ignore[arg-type]
                tail_block = entry_block(new_tail_entry, fold_width=fold)
                self._body.pin_tail(tail_block)
        elif new_tail_entry is not None:
            tail_block = entry_block(new_tail_entry, fold_width=fold)
            self._body.pin_tail(tail_block)
        elif old_tail_block is not None:
            self._body.remove_block(old_tail_block)
            tail_block = None

        self._blocks = [*body_blocks, *([tail_block] if tail_block is not None else [])]
        self._entries = list(entries)
        # Not while the initial history page is still in flight: that
        # prepend is the first glance a real child shows, and snapping
        # the trajectory-only body consumes the one-shot so sticky-tail
        # follow after the page lands bisects the wrapping notice again
        # (review round 1, F1). Comms-less fixtures never set the flag
        # and still snap here.
        if self._landing_snap_pending and not self._initial_tail_pending:
            # Two refreshes, not one: the first lays the blocks out, the
            # body's sticky-tail follow then moves the offset onto the
            # newest content (and, on a short viewport, onto a wrap
            # fragment). Snapping on the first refresh sees offset 0 and
            # would either no-op or consume the one-shot before the
            # fragment exists.
            self.call_after_refresh(self._schedule_landing_snap)

    def _schedule_landing_snap(self) -> None:
        if self._landing_snap_pending:
            self.call_after_refresh(self._snap_landing_to_row_head)

    def _snap_landing_to_row_head(self) -> None:
        """Land the first glance on a row HEAD, not a wrap fragment.

        Sticky-tail following puts the newest content at the bottom of the
        viewport. On a short terminal that can bisect a wrapping notice:
        the glyph and the start of the sentence sit above the fold, and
        the first visible line is a hanging-indented continuation in the
        notice's own red — a scrap that looks like a broken row rather
        than the rest of one (issue #407).

        The wrap itself is correct; only the landing is wrong. If the
        current offset sits strictly inside a block, pull it back to that
        block's start so the glyph (or the first line of prose) is the
        first thing on screen. Never past the tail — a snap that hid the
        working line would be a worse first glance than a fragment.

        One-shot on open. A reader who has already scrolled owns the
        offset; yanking it on a later refresh is the defect this is not.
        """
        if not self._landing_snap_pending:
            return
        # The trajectory-only body is not the first glance. Keep the
        # one-shot until `_settle_initial_landing` has the durable page
        # (or has learned there isn't one).
        if self._initial_tail_pending:
            return
        body = self._body
        if not body.is_mounted or not body.size.height:
            return
        # Layout has to have happened: a snap against a zero-height list
        # (the first refresh after mount, before Textual assigns regions)
        # would consume the one-shot and leave the wrap fragment in place.
        if not body.blocks() or body.virtual_size.height <= body.size.height:
            if body.max_scroll_y <= 0 and body.blocks():
                # Content fits: there is no fragment to snap off. Done.
                self._landing_snap_pending = False
            return
        offset = body.scroll_offset.y
        blocks = body.blocks()
        # Only decide when a block actually OWNS the current offset.
        #
        # The container republishes ``virtual_size`` (and so ``max_scroll_y``,
        # and so the followed offset) before every child's ``virtual_region``
        # has been reassigned, so there is a window where the offset addresses
        # a row no block claims yet. Searching in that window matches nothing,
        # falls through to ``target == offset``, and spends the one-shot on a
        # landing it never actually inspected.
        #
        # An unowned offset does NOT prove the layout is stale, and this test
        # is deliberately not written as if it did: the gap margin
        # (``.gap-above``) and the list's own vertical padding are real rows
        # that no block's region covers, so unowned offsets exist at steady
        # state too (measured: sets like ``[6, 8]`` in every configuration
        # tried). The condition being tested is narrower and is the one that
        # matters here — "is there a block whose head I could snap to?" With
        # no owner there is nothing to snap to, so there is no decision to
        # make and no reason to retire the guard.
        #
        # The cost of that is a one-shot which may never be spent on a
        # completed child that sends no further refresh. That is deliberate
        # and safe rather than merely tolerable: a surviving one-shot can only
        # ever snap to the head of the block that owns the CURRENT offset, so
        # firing it late is a no-op or a correction upward inside the block
        # the reader is already looking at — never a jump somewhere else.
        # Verified against the alternative during review: a reader parked
        # mid-block and then repainted is yanked in MORE configurations
        # without this guard than with it.
        #
        # Observed on CI four times with identical numbers — ``offset=28,
        # owner_top=25, max=28`` — and reproduced locally under xdist with the
        # trace ``snap-post 28 28 37 following=True pending=False``: the
        # deciding call saw the FINAL extent (37) and still found no owner,
        # because the notice's region had not been republished yet.
        #
        # Returning WITHOUT clearing the flag is the point: the one-shot
        # survives to the next refresh, which is what a one-shot is for. The
        # user-visible alternative is issue #407 itself — a narrow page
        # opening on a hanging-indented red continuation line instead of the
        # notice's first row.
        if not any(
            block.virtual_region.y <= offset < block.virtual_region.bottom for block in blocks
        ):
            return
        target = offset
        for block in blocks:
            top = block.virtual_region.y
            bottom = block.virtual_region.bottom
            if top < offset < bottom:
                target = top
                break
            if top >= offset:
                break
        cap = body.max_scroll_y
        if cap > 0:
            target = min(max(target, 0), cap)
        if target != offset:
            with body._tail_anchor.programmatic_scroll():
                body.scroll_to(y=target, animate=False, immediate=True)
        # Release whenever the landing is NOT the tail — including the case
        # where no scroll was needed because the offset already sat on a head.
        #
        # This release used to live inside the ``target != offset`` branch, so
        # a snap that found the offset already on a head spent the one-shot
        # while leaving sticky-follow armed. That is not a rare path: this
        # method runs several times during an open (measured: three on the
        # narrow fixture, the first two returning early against a body that is
        # still short), and any growth after the deciding call then sends
        # ``_size_updated`` -> ``_scroll_to_tail`` to the new tail. On a short
        # viewport that tail is three rows inside the wrapping notice — issue
        # #407 exactly, reached after the guard had declared the landing done.
        #
        # Measured as ``offset=28, owner_top=25, max=28`` on CI (four
        # identical failures) and reproduced locally at 1-in-8 under load.
        # Keyed on the POSITION rather than on whether a scroll happened,
        # because the invariant the anchor encodes is "the viewport is not at
        # the end", and that is equally true in both branches. Released AFTER
        # the programmatic context above: ``note_user_scroll`` is a no-op
        # while that depth is non-zero.
        if target < body.max_scroll_y:
            body._tail_anchor.release()
        self._landing_snap_pending = False

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
        # Counts come from the FOLDED rows the body already trusts, rather than
        # re-interpreting raw events in the chrome. Attempts remain visible while
        # settled errors add the outcome signal; an in-flight row has no outcome
        # yet and therefore must not be presented as a failure.
        tool_entries = [entry for entry in self._entries if entry.kind == "tool"]
        tools = len(tool_entries)
        failed_tools = sum(1 for entry in tool_entries if entry.outcome == "error")
        # The tool COUNTS, not the entry count: the memo has to key on what the
        # title renders, or a row settling from running to failed at equal length
        # paints a stale header. Sound on its own, rather than by way of the
        # spinner happening to repaint eight times a second.
        # Role and effort are part of what the title paints, so they belong in
        # the memo key: without them a page retargeted from a task to a scout
        # (same label width, different role) would keep the stale header.
        state = (
            self._label,
            self._status,
            self._queued,
            self._elapsed,
            tools,
            failed_tools,
            width,
            spinner,
            self._agent_role,
            self._effort,
            # The ANCESTORS belong here for the same reason role and effort do:
            # they are painted (by the breadcrumb) and nothing else in this key
            # implies them, so a page re-parented under a different lineage at
            # an unchanged label, status and width would be memoised out and
            # keep the stale trail. Cheap to carry — a tuple of at most a few
            # short strings, compared once per call.
            tuple(self._ancestors),
        )
        if self._chrome_state == state:
            return
        self._chrome_state = state
        # The theme is resolved at BUILD time into every ``Style`` below, so it
        # is a term of both per-row keys: without it a `/theme` switch under an
        # open page would be skipped by those keys and the breadcrumb and rule
        # would keep the old ramp's ink for the life of the page. The epoch
        # counter is this codebase's existing invalidation handle for exactly
        # that (see `assistant.py`'s frozen-epoch check); it is a module global
        # read, so carrying it costs nothing per tick.
        epoch = theme_mod.get_theme_epoch()
        self._title_text = self._title_row(width, spinner, tools, failed_tools)
        # Keyed on WIDTH alone: the row is a pure function of it, while the
        # memo above also carries the spinner and therefore fires eight times
        # a second — re-measuring five candidate rungs and layout-refreshing
        # four widgets for output that cannot have changed.
        if self._hint_width != width:
            self._hint_width = width
            self._hint_text = self._hint_row(width)
        # ``layout=False`` on all three: the sheet fixes each at ``height: 1``
        # (``.subagent-view-title``, ``.subagent-view-breadcrumb``,
        # ``.subagent-view-rule``) and each is built to the measured width, so
        # none can move the box. Textual's default reflows the screen, and the
        # memo above carries the spinner, so this runs 12.5 times a second for
        # as long as the child is alive. A/B in one process, 161 blocks behind
        # the page, three-second idle windows, two rounds: 4.4%/4.2% of a core
        # with the default against 3.5%/3.6% with this.
        #
        # The breadcrumb used to omit the keyword while its two siblings passed
        # it, which NEGATED both of theirs: one defaulted `update` three lines
        # below relayouts the same screen on the same tick, so the pair bought
        # nothing. Measured over 50 driven ticks with eight children and a
        # 160-block transcript, adding it here takes `messages.Layout` from
        # 1.08 per tick to 0.00 and compositor reflows from 160 to 32.
        self._title.update(self._title_text, layout=False)
        # Below this line each row carries its OWN key, because the memo above
        # is defeated on ~93% of ticks by design: `spinner` is one of its terms
        # and `_tick` advances the index before every call, so on a running
        # child the memo misses 291 times out of 313 in two seconds. The
        # spinner is painted by the TITLE and by nothing else — the breadcrumb
        # is a pure function of `_ancestors` + `_label`, the rule of width —
        # so without these keys both were rewritten with byte-identical strings
        # 12.5 times a second, each rewrite still costing a `messages.Update`
        # and a repaint of the row. Skipping the unchanged pair takes the
        # screen's Update messages from 48.5 to 30.6 per tick (-37%) and the
        # child's tick from 11.52 ms to 9.88 ms (-14.3%).
        #
        # This mirrors `SubagentPanel._tick`'s cheap glyph-only path rather
        # than inventing a second convention: there the panel repaints only the
        # rows carrying a glyph; here the page repaints only the row that
        # carries one. The animation's cadence is untouched — the title still
        # advances on every tick.
        breadcrumb_key = (tuple(self._ancestors), self._label, epoch)
        if self._breadcrumb_key != breadcrumb_key:
            self._breadcrumb_key = breadcrumb_key
            breadcrumb = "Conversation"
            if self._ancestors:
                breadcrumb += " > " + " > ".join(self._ancestors)
            breadcrumb += " > " + self._label
            self._breadcrumb.update(
                Text(breadcrumb, style=Style(color=theme_mod.semantic_color("dim"))),
                layout=False,
            )
        rule_key = (width, epoch)
        if self._rule_key != rule_key:
            self._rule_key = rule_key
            self._rule_text = Text(
                "─" * max(1, width - SCROLLBAR_GUTTER_CELLS),
                style=Style(color=theme_mod.semantic_color("faint")),
            )
            self._rule.update(self._rule_text, layout=False)

    def _title_row(self, width: int, spinner: str, tools: int, failed_tools: int = 0) -> Text:
        """Build the adaptive title, including attempted and failed tool counts.

        The breadcrumb is dim and the LABEL carries the base ink, which
        inverts the usage card's ``Usage  <target>`` weighting on purpose:
        there the noun is the subject and the target qualifies it, here the
        noun only says which surface you are on and the label is the title of
        the page being read — and it is the same string the band row beneath
        already paints at ``fg``.

        The ROLE rides the breadcrumb (``Subagent · scout · <label>``) because
        it is an identity fact about the page, read together with the surface
        name. The default ``task`` role is SUPPRESSED as noise — every child is
        a task unless told otherwise, so printing it says nothing — while a
        ``scout`` or any other non-default role is named. The EFFORT tier sits
        with the status group (``running · hi · 2m23s``): it qualifies the run
        the way the band's effort segment qualifies the model, and it is kept
        longer than the elapsed clock because the tier is part of why the page
        was opened, where the clock is the same value the band already carries.

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
        # The role rides the breadcrumb but is the MOST disposable field: it is
        # identity sugar the label already half-carries, so it is the first
        # thing to leave when the row tightens. It is offered ONLY at the
        # least-reduced rung — the full state word present and no tail field yet
        # dropped (``keep_word and dropped == 0`` below) — so it yields before
        # any status field, the clock, OR a shortened state word. Gating it on
        # that rung rather than merely trying it innermost is load-bearing: the
        # role chrome (``"scout · "``, 8 cells) is SHORTER than a long state
        # word (``" completed"``, 10 cells), so an innermost keep-role offer
        # would let a wordless-but-role-kept row win over a worded-but-roleless
        # one at ~30 cells — the exact invariant this field is documented to
        # hold, broken for ``completed``/``done`` where the word is long. The
        # default ``task`` role is never shown: every child is a task unless
        # told otherwise, so the word says nothing a reader did not assume.
        role_seg = self._agent_role if self._agent_role and self._agent_role != "task" else ""
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
                    attempts = f"{tools} tool{'' if tools == 1 else 's'}"
                    failures = f" · {failed_tools} failed" if failed_tools else ""
                    # Attempt and failure counts are one indivisible field: the
                    # existing whole-field ladder may keep both or drop both,
                    # but can never leave a clipped number that changes meaning.
                    tail.append((f" · {attempts}{failures}", dim))
                tail.append((f" · {self._elapsed}", dim))
                # Effort sits AFTER elapsed in the list, so it is dropped LATER
                # (the ladder peels ``tail`` from the front): the tier is part
                # of why the page was opened, where the clock only repeats the
                # band. It renders between the state word and the clock —
                # ``running · hi · 2m23s`` — because it qualifies the run the
                # way the band's effort segment qualifies the model.
                if self._effort:
                    tail.append((f" · {self._effort}", dim))
                tail.append((f" {word_choice}", dim))
                rungs = len(tail) if keep_word else len(tail) + 1
                for dropped in range(rungs):
                    fields = tail[dropped:]
                    # The role is the MOST disposable field on the row, so it is
                    # offered ONLY at the least-reduced rung: the full state word
                    # still present (``keep_word``) and no tail field yet dropped
                    # (``dropped == 0``). Every tighter rung tries the row
                    # WITHOUT the role, so the role leaves before a shortened
                    # state word, any dropped tail field, or the clock — the
                    # invariant it is documented to hold. Merely trying it
                    # innermost broke that for a long state word (see the
                    # ``role_seg`` comment above). Only offered when there is a
                    # non-default role to show.
                    role_allowed = bool(role_seg) and keep_word and dropped == 0
                    for keep_role in (True, False) if role_allowed else (False,):
                        role_chrome = cell_len(f"{role_seg} · ") if keep_role else 0
                        # The label gets whatever the fields do not want,
                        # floored at eight cells so it never vanishes entirely —
                        # it is the page's subject. The fixed chrome is the
                        # breadcrumb AND the glyph (and the role, when shown):
                        # counting only the 13 breadcrumb cells left the budget
                        # one short, so a rung whose label consumed it exactly
                        # was rejected and the ladder fell through — non-monotone
                        # in width, with the state word visible at 35 cells and
                        # gone again at 36-40 where it still fit.
                        spend = (
                            sum(cell_len(text) for text, _ in fields)
                            + 13
                            + cell_len(glyph)
                            + role_chrome
                        )
                        label = truncate_cells(self._label, max(8, width - spend))
                        row = Text(no_wrap=True, overflow="ellipsis")
                        row.append("Subagent", style=dim)
                        row.append(" · ", style=faint)
                        if keep_role:
                            row.append(role_seg, style=dim)
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

    def _navigate(self, relation: str) -> None:
        action = getattr(self.app, f"action_subagent_{relation}", None)
        if callable(action):
            action()

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
        relations = (self._parent_hint, self._peer_hint, self._child_hint, self._root_hint)
        rungs: tuple[tuple[tuple[HintButton, ...], str], ...] = (
            ((*relations, *arrows, self._exit_hint, self._state_hint), "back to conversation"),
            ((*arrows, self._exit_hint, self._state_hint), "back to conversation"),
            ((*relations, *arrows, self._exit_hint), "back to conversation"),
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
        for hint in (*relations, *arrows, self._exit_hint, self._state_hint):
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
        labels = {
            self._parent_hint: "parent",
            self._peer_hint: "peer",
            self._child_hint: "child",
            self._root_hint: "root",
            self._scroll_label: " scroll",
            self._exit_hint: esc_label,
        }
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
    def _spinner_interval(self) -> float:
        # Motion, not colour, is how this app says "alive", at the cadence the
        # band and the status line already use: two speeds on one screen read
        # as two different states. On a BLURRED terminal the three SPINNERS
        # (this page, the dock and the band) step together at the reduced rate,
        # so they still agree with each other.
        #
        # The working line is the deliberate exception and this comment used to
        # overclaim by omitting it: it falls to its pre-existing static mode,
        # where the glyph FREEZES and only the clock moves (design review D3).
        # That is the shimmer-off frame this app already ships and already
        # treats as legible, and a blurred screen therefore shows a still
        # working line beside two stepping spinners. Left as-is rather than
        # unified, because the alternative is inventing a fourth animation
        # state for a window nobody is looking at.
        return SPINNER_INTERVAL_S if animation_focused() else BLURRED_SPINNER_INTERVAL_S

    def _start_spinner(self) -> None:
        if self._spinner_timer is None:
            self._spinner_rate = self._spinner_interval()
            self._spinner_timer = self.set_interval(self._spinner_rate, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def sync_animation_rate(self) -> None:
        """Re-rate the spinner after a focus change.

        Textual's ``Timer`` has no public way to change its interval, so the
        timer is replaced rather than adjusted — which is safe because the
        spinner's phase is held in ``_spinner_index`` and not in the timer, so
        nothing about the animation resets. Only a page with a live timer is
        touched: re-rating a settled child would START a spinner it had
        correctly stopped.

        The chrome is repainted immediately on the way back to the fast rate so
        a refocused terminal shows the CURRENT glyph, status and clock rather
        than the frame it was throttled on — the reduced rate is allowed to
        cost frames, never to leave stale content on screen.
        """
        if self._spinner_timer is None:
            return
        wanted = self._spinner_interval()
        if wanted == self._spinner_rate:
            return
        self._stop_spinner()
        self._start_spinner()
        self._paint_chrome()

    def _tick(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        self._paint_chrome()


def _plain(renderable: Any) -> str:
    """A block's text, whatever kind of renderable it happens to hold."""
    plain = getattr(renderable, "plain", None)
    if isinstance(plain, str):
        return plain
    return str(renderable)
