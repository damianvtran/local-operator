"""Bounded prepared transcript presentations, independent of owner attachment.

The owner and its canonical state stay authoritative. This module only owns
widgets and replay bookkeeping; preparing one has no subscription, prompt,
acknowledgement, or reference to the currently selected app session.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from textual import events
from textual.binding import Binding
from textual.message import Message

from local_operator.harness.types import ImageContent
from local_operator.tui.session_interaction import SessionDraft
from local_operator.tui.widgets.image_block import ImageBlock
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptBlock,
    TranscriptView,
)

#: Resident bytes of retained text one parked presentation may hold, measured
#: with ``sys.getsizeof`` rather than estimated from a character count. It must
#: stay >= ``DISPLAY_HISTORY_BYTES`` (512 KiB): the window layer is permitted to
#: produce a payload that large, so a smaller retain budget structurally refuses
#: presentations the layer above legitimately built. See
#: :meth:`SessionPresentation.retainable` for what this protects against and how
#: to measure a real presentation against it before changing it.
RETAIN_TEXT_BYTES = 1024 * 1024


class HistoryPageNotice(NoticeBlock, can_focus=True):
    BINDINGS = [Binding("enter", "more", "More recent messages", show=False)]

    class Requested(Message):
        def __init__(self, notice: HistoryPageNotice) -> None:
            super().__init__()
            self.notice = notice

    def __init__(self) -> None:
        super().__init__("More recent messages below", "note")
        self.add_class("interactive-notice")

    def action_more(self) -> None:
        self.post_message(self.Requested(self))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_more()


class OlderHistoryNotice(NoticeBlock, can_focus=True):
    """The HEAD notice, and the twin of :class:`HistoryPageNotice` above it.

    The two ends of one transcript solve the same problem — "there is more
    conversation off this edge of the screen" — and they must not solve it in
    opposite ways. The tail notice has always been a control: focusable,
    clickable, `enter`-bound, with the `interactive-notice` styling that gives
    it hover and focus affordances. The head notice was a bare label that
    recited a keyboard chord (`ctrl+home`) which appears NOWHERE else in the
    product — not in the footer hints, not in the `?` help screen — and which
    on most Mac keyboards is itself a chord (`fn+ctrl+←`). A reader who had
    learned to click the bottom notice would click this one and get nothing.

    Making it a control is what lets the copy stay a plain statement: the row
    no longer has to explain how to operate it, because it IS the thing you
    operate. That matters most in the state this notice exists for — a frame
    too tall to scroll, where the history is otherwise a dead end.
    """

    BINDINGS = [Binding("enter", "older", "Older messages", show=False)]

    class Requested(Message):
        def __init__(self, notice: OlderHistoryNotice) -> None:
            super().__init__()
            self.notice = notice

    def __init__(self, text: str) -> None:
        super().__init__(text, "note")
        self._interactive = True
        self.add_class("interactive-notice")

    def set_interactive(self, interactive: bool) -> None:
        """Advertise an action only while there is one to take.

        The head notice restates rather than removing itself when the history
        runs out (removing the first row would shift every row below it and
        undo the anchor an insert just held), so unlike its tail twin it
        outlives its own action. A row that keeps `interactive-notice`, keeps
        `can_focus`, and paints the full-width focus band while activating it
        does nothing is a focus stop that answers Enter with silence.

        `can_focus` is an instance attribute here, shadowing the class-level
        value Textual's `can_focus=True` keyword set. Textual reads
        `allow_focus()` -> `can_focus` per widget at focus time, so flipping it
        removes the row from the focus chain without touching the class or the
        twin.

        Idempotent, and reversible in both directions: a remote page can refill
        an exhausted head, and a control that only ever went one way would be
        the same stale-state defect the copy already guards against.
        """
        if interactive == self._interactive:
            return
        self._interactive = interactive
        if not interactive and self.has_focus:
            # Focus cannot rest on a row that just left the focus chain: it
            # would keep the band painted and keep swallowing Enter.
            #
            # ORDER IS LOAD-BEARING: blur BEFORE `can_focus` is cleared.
            # `blur()` -> `Screen._reset_focus(self)` locates this widget in
            # the focus chain to hand focus to an ordered NEIGHBOUR. Clearing
            # `can_focus` first removes the row from that chain, so the lookup
            # raises and Textual takes its "widget was made invisible" fallback
            # instead: the first focusable VISIBLE SIBLING, which in a
            # transcript is the topmost `ToolCard` — measured ~770 rows above a
            # reader sitting at the tail, with the reader's next `enter`
            # silently expanding a card they cannot see (review round 3, R9).
            # Blurring while still in the chain lands focus on `TranscriptView`,
            # which is on screen and answers `enter` with nothing.
            self.blur()
        self.can_focus = interactive
        self.set_class(interactive, "interactive-notice")

    def action_older(self) -> None:
        self.post_message(self.Requested(self))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        if not self._interactive:
            # Stopped anyway: the row still occupies its cells, and letting the
            # click fall through to the transcript beneath would scroll a
            # surface the reader was pointing at, not aiming past.
            return
        self.action_older()


class DraftRecoveryNotice(NoticeBlock, can_focus=True):
    BINDINGS = [Binding("enter", "restore", "Restore unsent prompt", show=False)]

    class Requested(Message):
        def __init__(self, notice: DraftRecoveryNotice) -> None:
            super().__init__()
            self.notice = notice

    def __init__(self, source_token: str, draft: SessionDraft) -> None:
        super().__init__("Restore unsent prompt", "warning")
        self.source_token = source_token
        self.draft = draft
        self.add_class("interactive-notice")

    def action_restore(self) -> None:
        self.post_message(self.Requested(self))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_restore()


@dataclass
class ReplayState:
    _resume_results: dict[str, Any] = field(default_factory=dict)
    _resume_pending_head: list[Any] = field(default_factory=list)
    _resume_pending_tail: list[Any] = field(default_factory=list)
    _resume_tail_notice: NoticeBlock | None = None
    _resume_head_notice: NoticeBlock | None = None
    _resume_mounted_ids: set[str] = field(default_factory=set)
    _replay_bang_pending: bool = False
    _live_peer_receipts: set[str] = field(default_factory=set)
    _live_wake_receipts: set[tuple[str, object]] = field(default_factory=set)
    _block_sink: list[Any] | None = None
    _projection_message_id: str = ""
    _projection_part: int = 0


class ReplayTarget(Protocol):
    _resume_results: dict[str, Any]
    _resume_pending_head: list[Any]
    _resume_pending_tail: list[Any]
    _resume_tail_notice: NoticeBlock | None
    _resume_head_notice: NoticeBlock | None
    _resume_mounted_ids: set[str]
    _replay_bang_pending: bool
    _live_peer_receipts: set[str]
    _live_wake_receipts: set[tuple[str, object]]
    _block_sink: list[Any] | None
    _projection_message_id: str
    _projection_part: int

    def _transcript_view(self) -> TranscriptView: ...

    def _append_block(
        self, block: Any, *, ends_empty_state: bool = True, pin_tail: bool = False
    ) -> None: ...

    def _append_image_blocks(
        self, images: list[ImageContent], *, marker_text: str | None = None
    ) -> list[ImageBlock]: ...

    def _painted_tool_card(self, call_id: str) -> Any: ...

    def _settle_painted_tool_card(self, card: Any, result: Any) -> None: ...

    def _replay_tool_call(
        self, call: Any, results: dict[str, Any], *, user_run: bool = False
    ) -> None: ...


@dataclass
class PreparedReplay(ReplayState):
    _resume_head_notice: NoticeBlock | None = None
    _resume_tail_notice: NoticeBlock | None = None
    view: TranscriptView = field(default_factory=TranscriptView)
    blocks: list[TranscriptBlock] = field(default_factory=list)

    def _transcript_view(self) -> TranscriptView:
        return self.view

    def _append_block(
        self, block: Any, *, ends_empty_state: bool = True, pin_tail: bool = False
    ) -> None:
        # Same recording as the live appender: a prepared replay decides its own
        # empty state from these blocks (`prepare` below), and the commit path
        # asks the mounted view the same question afterwards.
        block.ends_empty_state = ends_empty_state
        if not block.navigation_anchor_id:
            block.navigation_anchor_id = self._projection_message_id
            block.navigation_anchor_part = self._projection_part
        self._projection_part += 1
        self.blocks.append(block)

    def _append_image_blocks(
        self, images: list[ImageContent], *, marker_text: str | None = None
    ) -> list[ImageBlock]:
        return append_image_blocks(self, images, marker_text=marker_text, navigation_visible=False)

    def _painted_tool_card(self, call_id: str) -> None:
        return None

    def _settle_painted_tool_card(self, card: Any, result: Any) -> None:
        raise AssertionError("a prepared replay cannot contain a live tool card")

    def _replay_tool_call(
        self, call: Any, results: dict[str, Any], *, user_run: bool = False
    ) -> None:
        replay_tool_call(self, call, results, user_run=user_run)

    def prepare(self, history: list[Any], *, bound: int = 12, anchor_id: str = "") -> None:
        self._block_sink = self.blocks
        anchor = (
            next(
                (
                    index
                    for index, message in enumerate(history)
                    if str(getattr(message, "id", "")) == anchor_id
                    or any(
                        f"tool:{getattr(call, 'id', '')}" == anchor_id
                        for call in getattr(message, "tool_calls", ())
                    )
                ),
                None,
            )
            if anchor_id
            else None
        )
        end = min(len(history), anchor + bound) if anchor is not None else None
        project_settled_rows(self, history, bound=bound, end=end)
        if self._resume_pending_head:
            from local_operator.tui.app import RESUME_OLDER_NOTICE

            self._resume_head_notice = OlderHistoryNotice(RESUME_OLDER_NOTICE)
            self.blocks.insert(0, self._resume_head_notice)
        if self._resume_pending_tail:
            self._resume_tail_notice = HistoryPageNotice()
            self.blocks.append(self._resume_tail_notice)
        self._block_sink = None


@dataclass
class SessionPresentation:
    replay: PreparedReplay
    revision: int = 0
    replay_revision: int = 0
    source_stamp: tuple[Any, ...] = ()
    source_token: str = ""
    history_size: int = 0
    needs_live_projection: bool = True
    streaming_block: Any = None
    tool_cards: dict[str, Any] = field(default_factory=dict)
    composing_cards: dict[str, Any] = field(default_factory=dict)
    working_block: Any = None
    working_fallback: str = ""
    compaction_owns_working_block: bool = False
    shell_card: Any = None
    queued_steer_notices: list[Any] = field(default_factory=list)
    deferred_steer_notices: list[Any] = field(default_factory=list)
    held_steer_blocks: list[Any] = field(default_factory=list)
    welcome: Any = None
    welcome_visible: bool | None = False

    def retainable(self) -> bool:
        """Bound explicit payloads, not a widget's private object graph.

        One cached view can otherwise retain arbitrarily much history. Unknown
        renderers (including decoded images) are rebuilt instead of guessing.

        **What the budget is protecting against, and in what units.** The bound
        is RESIDENT BYTES of retained text, not characters and not the JSON
        wire frame the window layer prices itself in. It exists so ONE parked
        view cannot pin an unbounded slice of a 200 MB journal in RAM;
        ``RETAINED_PRESENTATIONS`` parked views each get this budget, so
        N x this number bounds the retained TEXT.

        It does **not** bound a retained presentation's total resident cost.
        The mounted ``TranscriptView`` and its widget tree are parked (offset
        ``100vw``), not freed, and that tree is not what this method measures —
        so N x the budget is a bound on one term, not a memory ceiling for the
        cache. Stated plainly because the previous wording ("the real ceiling
        is N x this number") reads as the latter, and the next person tuning
        this needs to know which quantity they are holding. The text term is
        still the one worth bounding here: it is the term that scales with
        conversation length, which is what makes a long transcript expensive.

        **Why ``sys.getsizeof`` and not ``len(value) * k``.** CPython stores
        ``str`` in PEP 393 compact form: 1, 2 or 4 bytes per character
        depending on the widest code point, plus a header. So no constant
        multiplier is right for all content — measured here, the same 4096
        characters cost 4137 bytes as ASCII, 8250 as BMP and 16444 as astral.
        ``getsizeof`` is the honest measure of one string object, and it costs
        ~75 ns against ~29 ns for ``len`` on a 500 KB string — irrelevant
        beside the mount + layout this predicate decides whether to repeat.

        It is honest **per object**, which is why the charge sits below the
        ``id(value) in seen`` check: charged above it, one string aliased by N
        blocks was charged N times for a single allocation (measured: one
        300 KB string under 5 keys charged 1.43 MiB and was refused at a true
        cost of 0.29 MiB). Over-charging is how both previous mis-tunings
        failed, so the identity check is load-bearing, not tidiness.

        **This bound has now been mis-tuned twice, in the same direction.**
        The node cap (see the comment below) silently refused every real owner
        once. Then the string term charged ``len(value) * 4`` — a worst-case
        UTF-32 assumption — against a 1 MiB budget, so the effective budget was
        256 KiB while ``DISPLAY_HISTORY_BYTES`` permits the window layer to
        hand this predicate a 512 KiB payload. The two limits contradicted each
        other and the cache silently never cached: a refused presentation is
        never inserted into ``_sidebar_presentations``, so it never stops
        matching the prewarm candidate filter and is fully re-prepared (mount +
        layout + unmount, on the event loop) on every 2 s sidebar poll. Three
        of the operator's ten live sessions were stuck in that loop, which is
        the reported "lag with the sidebar open, fine when I close it".

        **How to verify this rather than re-derive it.** Do not reason about
        the multiplier; measure a real presentation. Walk the same roots this
        method walks, sum ``sys.getsizeof`` over the strings, and compare
        against ``RETAIN_TEXT_BYTES``. On the operator's eight largest real
        transcripts the retained text measures 3-660 KiB (the roots are the
        blocks plus the unmounted ``_resume_pending_head``/``_resume_results``
        paging buffers, which dominate at 127-357 KiB), so 1 MiB admits every
        one of them with the largest at 64% of budget. If you are considering
        tightening this, get that measurement first: the failure mode of a too-
        tight bound here is not a rejected cache entry, it is a permanent
        re-preparation loop that looks like general UI slowness.
        """
        if len(self.replay.view.blocks()) > 128:
            return False
        state = self.replay
        stack: list[Any] = [
            self.replay.view.blocks(),
            state._resume_pending_head,
            state._resume_pending_tail,
            state._resume_results,
            self.tool_cards,
            self.composing_cards,
            self.streaming_block,
            self.working_block,
            self.shell_card,
            self.queued_steer_notices,
            self.deferred_steer_notices,
            self.held_steer_blocks,
        ]
        seen: set[int] = set()
        # `seen` stores ADDRESSES, and an address only identifies an object
        # among those that are simultaneously ALIVE. Every node the walk drops
        # can therefore have its address handed to a later, unrelated node,
        # which `seen` then skips as already-visited — silently un-charging it.
        # This is not hypothetical: `retained_payloads()` builds a FRESH tuple
        # per block, so on a 64-block presentation holding 64 KiB of distinct
        # text each, 62 of 64 tuples reused a freed address and the walk
        # charged 0.13 MiB against 4.00 MiB actual — and admitted it.
        #
        # So everything that gets an id in `seen` is kept alive here until the
        # walk returns. Bounded by construction: the node cap caps the pointer
        # count, and the string budget caps the transient TEXT held, because
        # the walk returns False as soon as `remaining` goes negative.
        alive: list[Any] = []
        remaining = RETAIN_TEXT_BYTES
        nodes = 0
        while stack:
            value = stack.pop()
            nodes += 1
            # The node cap guards against a pathological object graph, not
            # memory — the 1 MiB string budget below is the memory bound. At
            # 4096 it silently refused every real owner: a pydantic Message
            # costs ~14 nodes, so a kept window past ~290 messages (the live
            # owners hold 355–456) was never retained and every return click
            # was cold (retained_views stayed at 1 with N=4 configured).
            # 65536 admits ~4,600 plain messages, which the string budget
            # trips long before on any real content.
            if nodes > 65536:
                return False
            if value is None or isinstance(value, (bool, int, float)):
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            alive.append(value)
            if isinstance(value, str):
                # Resident cost, not character count: see the docstring. A
                # `len(value) * 4` estimate here over-charged ASCII content 4x
                # and disabled the cache entirely.
                #
                # Charged BELOW the identity check, so one string object held
                # by N blocks costs one copy of RAM and is charged once. Above
                # it, a transcript with repeated identical tool output was
                # charged N x for a single allocation — an over-estimate, and
                # over-estimating is the direction that produced both previous
                # mis-tunings.
                remaining -= sys.getsizeof(value)
            elif isinstance(value, bytes):
                remaining -= len(value)
            else:
                if isinstance(value, TranscriptBlock):
                    payload = value.retained_payloads()
                    if payload is None:
                        return False
                    stack.append(payload)
                elif isinstance(value, BaseModel):
                    stack.extend(getattr(value, key) for key in type(value).model_fields)
                elif isinstance(value, Mapping):
                    if len(value) + len(stack) > 4096:
                        return False
                    stack.extend(value.values())
                elif isinstance(value, (list, tuple, set)):
                    if len(value) + len(stack) > 4096:
                        return False
                    stack.extend(value)
                else:
                    return False
            if remaining < 0:
                return False
        return True


def project_settled_rows(
    self: ReplayTarget, history: list[Any], *, bound: int | None = None, end: int | None = None
) -> bool:
    """Mount settled transcript rows through the ONE role-aware renderer.

    The shared history/render seam: cold resume feeds it the whole
    conversation and a reconnect's durable gap replay
    (:class:`HistoryRowsSettled`) feeds it exactly the rows no frontend
    painted. One implementation is the point — the gap replay previously
    synthesized role-blind assistant events and a recovered user prompt
    painted as agent speech (review round 3, MAJOR-1/U7/D1). Whatever
    this method does for ``--resume`` is by construction what a
    reconnect gap does: user rows as :class:`UserBlock` with images,
    assistant prose + tool cards paired with results, wake/peer custom
    rows as their own blocks, refusal/error notices.

    Returns whether anything mounted, so callers can skip tail-follow
    work for an empty projection.

    ``bound`` renders only the LAST ``bound`` messages and holds the rest
    for :meth:`_mount_older_resume_page`. It is a display bound and nothing
    else: the deferred messages stay in ``_resume_pending_head`` in full,
    and the model's conversation — built from the transcript, not from this
    projection — never sees the split at all. The gap-replay caller passes
    no bound, because a reconnect gap is by definition the small set of
    rows no frontend painted and bounding it could hide one.
    """
    from contextlib import nullcontext

    from local_operator.compaction.marker import COMPACTION_REFUSED_TYPE
    from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
    from local_operator.tui.app import (
        LOOP_PROMPT,
        PEER_MESSAGE_MESSAGE_TYPE,
        RESUME_OLDER_NOTICE,
        WAKE_PROMPT_MESSAGE_TYPE,
        AssistantBlock,
        PeerMessageBlock,
        UserBlock,
        WakeBlock,
        _gate_timeout_notice,
        _resume_tail_start,
        _typed_line_of,
    )

    # Results are keyed by the call they answer, and a tool message can sit
    # several messages after its call (one assistant turn issues a batch).
    # Indexing first is what lets each call render WITH its outcome instead
    # of as a second, orphaned row.
    #
    # Indexed over the WHOLE history, before any bound is applied: a call in
    # the deferred head is often answered by a result inside the rendered
    # tail, and a per-page index would show that call as `interrupted`.
    #
    # Seeded from the whole-conversation index a bounded resume kept, so a
    # deferred page's call still finds a result that lives in the already
    # rendered tail. Empty for every unbounded caller.
    results: dict[str, Any] = dict(self._resume_results)
    settled_results: set[str] = set()
    # Harness chrome the LIVE path never paints as a user row, so replay
    # must not either (see the `role == "user"` branch). Deferred once to
    # the top of this method rather than inside the loop, matching the
    # file's other lazy session.* imports.
    from local_operator.harness.loop import CONNECTIVITY_CONTINUATION_PROMPT
    from local_operator.session.session import _CONTINUATION_PROMPT

    for message in history:
        if getattr(message, "role", None) != "tool":
            continue
        call_id = getattr(message, "tool_call_id", None)
        if not call_id:
            continue
        results[call_id] = message
        # A result whose call painted LIVE before a disconnect must
        # settle the card already on screen — the disconnect marked it
        # ``interrupted``, and replaying it as a new row would double the
        # card (review round 4, MINOR-1). The disconnect retired the card
        # out of ``_tool_cards`` but left it mounted, so scan the
        # transcript for the painted card carrying this call id.
        painted = self._painted_tool_card(call_id)
        if painted is not None:
            self._settle_painted_tool_card(painted, message)
            settled_results.add(call_id)
    # Fresh batch, fresh pairing: a flag left by an earlier replay (or a
    # truncated one) must not open a card in this conversation.
    self._replay_bang_pending = False

    # Split the conversation into the head this frame defers and the tail it
    # paints. Sliced on MESSAGES rather than on rendered blocks because the
    # split has to be decided before anything is built — deciding it by
    # block count would mean building the blocks first, which is the cost
    # being avoided. A message mounts 0-2 blocks, so the block count lands
    # near the bound rather than on it, which is fine: the bound is a budget,
    # not a contract about how many rows appear.
    if end is not None and end < len(history):
        # Pair results against the whole history, then retain only the chosen
        # viewport window. Newer rows remain reachable through forward paging.
        self._resume_results = results
        self._resume_pending_tail = history[end:]
        history = history[:end]
    if bound is not None and len(history) > bound:
        start = _resume_tail_start(history, bound)
        if start > 0:
            deferred, history = history[:start], history[start:]
            # Whole-conversation results, so a deferred call still pairs with a
            # result that renders (or already rendered) in the tail.
            self._resume_results = results
            self._resume_pending_head = deferred
    transcript = self._transcript_view()

    appended = bool(settled_results)
    # The "older messages" notice has to be the FIRST row, so it is
    # appended before the batch rather than prepended after it.
    # `prepend_blocks` restores a scroll anchor on a later refresh, which
    # would fight `follow_tail` for the same frame — the reflow-after-paint
    # #451/#452 exist to prevent. One extra mount of a one-line notice is
    # not the cost this bound is avoiding.
    if self._block_sink is None and self._resume_pending_head and self._resume_head_notice is None:
        notice = OlderHistoryNotice(RESUME_OLDER_NOTICE)
        self._resume_head_notice = notice
        self._append_block(notice)
        appended = True
    # ONE mount for the whole conversation. Per-block mounting made Textual
    # re-walk its stylesheet, invalidate the container and schedule a settle
    # callback 297 times over on a 396-message session, for a layout that is
    # only looked at once — see `TranscriptView.batch_append`. A collected
    # backward page must not open a batch on the live transcript: the
    # blocks are inserted later, and an empty batch still schedules a
    # settle pass that would race the insert's own settle.
    batch = nullcontext() if self._block_sink is not None else transcript.batch_append()
    with batch:
        for message in history:
            self._projection_message_id = str(getattr(message, "id", ""))
            self._projection_part = 0
            # A wake delivery is a CustomMessage, so it has no ``role``
            # and would fall through every branch below — which is exactly
            # why a resumed session showed the agent answering a wake with
            # no sign the wake ever fired. Replaying it as its own block
            # keeps the receipt on screen. The catch-up prompt is
            # user-attributed, so replaying it too would put a raw
            # '(alarm) The session resumed…' line in the transcript as if
            # the user had typed it.
            if getattr(message, "custom_type", None) == WAKE_PROMPT_MESSAGE_TYPE:
                details = getattr(message, "details", None) or {}
                if not details.get("wake_catchup"):
                    key = (str(details.get("wake_id", "")), details.get("occurrence"))
                    # Skip a receipt this session already painted live —
                    # replaying it would double the line (round 2, m2).
                    if key not in self._live_wake_receipts:
                        self._append_block(WakeBlock(str(details.get("text", "")), catchup=False))
                        appended = True
                continue
            # A peer message (`lop send` from another session) is also a
            # CustomMessage with no ``role`` and would otherwise fall
            # through, leaving a resumed session with the agent's reply but
            # no sign the peer note arrived. Replay it as its own block,
            # skipping one already painted live this session (double-paint
            # guard, mirroring the wake branch above).
            if getattr(message, "custom_type", None) == PEER_MESSAGE_MESSAGE_TYPE:
                details = getattr(message, "details", None) or {}
                if str(getattr(message, "id", "")) not in self._live_peer_receipts:
                    self._append_block(
                        PeerMessageBlock(
                            str(details.get("body", "")),
                            details.get("sender") or {},
                        )
                    )
                    appended = True
                continue
            # A gate that timed out unattended is the most expensive event
            # in the detached feature — up to a day of held residency ends
            # here — and it rendered NOWHERE (round 1, D2/U2): the user
            # returned to a conversation that promised an action and
            # appeared to simply stop. The payload already carried the
            # tool, the description and the wait; only a renderer was
            # missing.
            #
            # `warning` ink because it is a state the user must know about,
            # not a receipt they can skip: a tool was denied, and denied by
            # expiry rather than by their decision — which is the same
            # distinction the transcript row itself exists to preserve.
            if getattr(message, "custom_type", None) == GATE_TIMEOUT_CUSTOM_TYPE:
                details = getattr(message, "details", None) or {}
                self._append_block(NoticeBlock(_gate_timeout_notice(details), kind="warning"))
                appended = True
                continue
            # A compaction that did NOT run. Rendered here for the same
            # reason as the row above: a custom row with no renderer is a
            # row nobody sees, and this one exists to CORRECT the
            # optimistic "compacting context…" receipt the routed command
            # already showed (round 5, U17). `warning` ink because the
            # context the user asked to reclaim is still there.
            if getattr(message, "custom_type", None) == COMPACTION_REFUSED_TYPE:
                details = getattr(message, "details", None) or {}
                text = str(details.get("detail") or "compaction did not run").strip()
                kind = "error" if text.startswith("compaction failed") else "warning"
                self._append_block(NoticeBlock(text, kind=kind))
                appended = True
                continue
            role = getattr(message, "role", None)
            if role == "tool":
                continue  # already rendered beside the call that asked for it
            text = getattr(message, "text", "") or ""
            text = text.strip() if isinstance(text, str) else ""
            if role == "user":
                # The live path never paints harness chrome as a user row
                # (LOOP_PROMPT is registered as a pending echo and consumed;
                # the auto-continuation prompt is never announced at all).
                # Replay must make the same choice, or a resumed session
                # shows rows the live one deliberately suppressed — the
                # live/replay divergence review round 2 pinned.
                #
                # The network-continuation prompt joins them for the same
                # reason: it is persisted so the TRANSCRIPT explains why one
                # answer arrived in two pieces, but the user never typed it
                # and the live run showed a NoticeEvent instead.
                if text in (
                    LOOP_PROMPT,
                    _CONTINUATION_PROMPT,
                    CONNECTIVITY_CONTINUATION_PROMPT,
                ):
                    continue
                # A `$skill` invocation persists as its EXPANDED payload,
                # because that is what the model was sent. Replaying it
                # verbatim showed a resumed conversation the whole SKILL.md
                # body as the user's row — and, since the picker titles a
                # session from its first user turn, named every such thread
                # "The user invoked the `research` skill…". The typed line
                # rides the payload's own opening tag, so replay repaints
                # exactly what the live session painted. Same live/replay
                # parity rule as the two prompts skipped above.
                text = _typed_line_of(text) or text
                # The images ride the persisted message as base64 content
                # blocks — the same bytes the model saw — so a resumed
                # prompt replays WITH its pictures, not just the receipt
                # count. This is the resume half of the promise the live
                # path makes in `_submit_prompt`.
                replay_images = [
                    block
                    for block in (getattr(message, "content", None) or [])
                    if isinstance(block, ImageContent)
                ]
                if text or replay_images:
                    self._append_block(UserBlock(text, len(replay_images)))
                    self._append_image_blocks(replay_images, marker_text=text)
                    appended = True
                # A bang-mode receipt replays as open as it lived: the
                # user row is `! <command>` and the assistant message that
                # follows carries exactly one bash call. Remembered so the
                # call's card can open on settle, the same contract the
                # live path makes.
                if text.startswith("! "):
                    self._replay_bang_pending = True
                continue
            if role != "assistant":
                continue
            # Consume the pending bang marker on EVERY assistant message:
            # record_shell writes the call-bearing assistant immediately
            # after the `!` row, and a later unrelated turn must never
            # inherit the open-on-settle flag.
            bang_pending = self._replay_bang_pending
            self._replay_bang_pending = False
            if text:
                block = AssistantBlock()
                block.completion_anchor_id = str(getattr(message, "id", ""))
                block.update_text(text)
                block.finalize_text()
                self._append_block(block)
                appended = True
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                # Only the FIRST call of a bang assistant message is the
                # command's own card; the shape record_shell writes has
                # exactly one, so consuming here is exact in practice and
                # conservative in theory.
                user_run = bool(
                    bang_pending and tool_calls[0] is call and getattr(call, "name", "") == "bash"
                )
                self._replay_tool_call(call, results, user_run=user_run)
                appended = True
            stop = getattr(message, "stop_reason", None)
            if stop == "refusal":
                # A refused turn replays its refusal even when the model DID
                # stream some prose first (Gemini safety stops often cut a
                # partial answer): the prose alone reads as a complete,
                # oddly short reply, and the user re-reading the session
                # needs to know the provider cut it off and why. The message
                # itself was stashed on the assistant message by the loop
                # precisely so this replay could show it.
                payload = getattr(message, "provider_payload", None) or {}
                # The fallback keeps the marker grammar (D3): every other
                # refusal line ends in a parenthetical, and a user who has
                # learned that shape would read its absence as meaningful.
                refusal = str(payload.get("refusal") or "") or (
                    "model refused the request (no details recorded)"
                )
                self._append_block(NoticeBlock(refusal, "error"))
                appended = True
            elif not text and not tool_calls:
                # An assistant message with neither prose nor a call is a
                # turn that FAILED. Skipping it is what left a resumed
                # session showing a prompt and nothing after it, with no
                # hint that the answer had errored rather than never been
                # asked for.
                if stop in ("error", "aborted"):
                    reason = "turn failed" if stop == "error" else "interrupted"
                    self._append_block(NoticeBlock(reason, "error"))
                    appended = True
    # Every message this pass rendered, by stable id — the dedupe key a
    # later backward page is filtered through.
    self._projection_message_id = ""
    self._resume_mounted_ids.update(
        str(getattr(message, "id", "")) for message in history if getattr(message, "id", None)
    )
    if self._block_sink is not None:
        # A collected page mounts nothing and owns no viewport: the head
        # notice belongs to the first render, and `follow_tail` would drag
        # the reader from the history they scrolled up to read down to the
        # newest turn — the exact opposite of the gesture that asked for it.
        return appended
    if appended:
        # Replay is mounted as one synchronous batch, before Textual can
        # remeasure the growing container between blocks. Land the reader on
        # the latest turn and ARM the anchor there, so the first thing the
        # resumed session streams carries them with it rather than growing
        # off the bottom of a viewport pinned to the replay's last frame.
        transcript.follow_tail()
    return appended


def replay_tool_call(
    self: ReplayTarget, call: Any, results: dict[str, Any], *, user_run: bool = False
) -> None:
    """Mount one settled tool row for a call from a previous session.

    The card is built exactly as a live one is — same constructor, same
    summary derivation from the arguments — so a resumed row is
    indistinguishable from the row the user watched run, apart from the
    duration the transcript never recorded.
    """
    from local_operator.tui.app import ImageContent, ToolCard, _first_line

    card = ToolCard(
        getattr(call, "id", "") or "",
        getattr(call, "name", "") or "",
        getattr(call, "arguments", None) or {},
        user_run=user_run,
    )
    self._append_block(card)
    result = results.get(getattr(call, "id", "") or "")
    if result is None:
        # No result recorded: the session ended between the call and its
        # answer. Showing it as complete would invent an outcome.
        card.restore(state="interrupted")
        return
    result_text = getattr(result, "text", "") or ""
    payload = getattr(result, "provider_payload", None) or {}
    details = payload.get("details") if isinstance(payload, dict) else None
    if getattr(result, "is_error", False) and result_text.startswith("aborted ("):
        # A user-stopped bang command persists as an error result (the
        # model-facing shape), but the LIVE frame it came from was the dim
        # shut `interrupted ⊘` row. Replaying it through the error branch
        # would reopen the user's own Esc as a red failure (design round
        # 1, D1). The aborted prefix is execute_bash's stable contract.
        card.restore(state="interrupted")
        return
    if getattr(result, "is_error", False):
        card.restore(
            state="error",
            result_text=result_text,
            details=details,
            error=_first_line(result_text),
        )
    else:
        card.restore(state="success", result_text=result_text, details=details)
    # Same rule as `on_tool_ended`: a result carrying image blocks shows
    # them under the settled card, so a resumed session's screenshots are
    # back on screen exactly where the live session showed them.
    self._append_image_blocks(
        [
            block
            for block in (getattr(result, "content", None) or [])
            if isinstance(block, ImageContent)
        ]
    )


def append_image_blocks(
    self: ReplayTarget,
    images: list[ImageContent],
    *,
    marker_text: str | None = None,
    navigation_visible: bool = True,
) -> list[ImageBlock]:
    """Mount one :class:`ImageBlock` per image, in order.

    The single entry point for putting pictures on the transcript — the
    prompt path, the tool-result path, and the resume replay all route
    here so a rendering decision (caps, protocol, the unavailable
    receipt) is made in exactly one place. Guarded per block: a block
    whose bytes will not decode still mounts (as its unavailable
    receipt), but a failure CONSTRUCTING one must not take down the
    message dispatch that carried a perfectly good tool result.

    Labels name WHICH of several images a receipt is about, and only
    when the batch has more than one — a receipt for a batch of one
    names nothing the row above it has not already said (review round
    1, F4). Where ``marker_text`` is given (the prompt paths), the
    numbers are read from the text's own ``[Image #N]`` citations, in
    citation order — the same walk ``resolve_markers`` built ``images``
    from — because marker numbers are not positional: delete #1 and
    paste again and the draft reads ``[Image #2] [Image #3]``, so a
    positional ``#1`` would name a marker the prompt does not contain
    (review round 2, F9). Tool results have no markers and fall back to
    positions.
    """
    from local_operator.tui.app import logger

    indices: list[int] = []
    if marker_text:
        from local_operator.tui.widgets.editor import IMAGE_MARKER

        indices = [int(match.group("index")) for match in IMAGE_MARKER.finditer(marker_text)]
    mounted: list[ImageBlock] = []
    for index, image in enumerate(images):
        if len(images) <= 1:
            label = ""
        elif index < len(indices):
            label = f"#{indices[index]}"
        else:
            label = f"#{index + 1}"
        try:
            block = ImageBlock(
                image.data or None,
                image.mime_type,
                label=label,
                navigation_visible=navigation_visible,
            )
        except Exception:
            logger.debug("image block construction failed", exc_info=True)
            continue
        self._append_block(block)
        mounted.append(block)
    return mounted
