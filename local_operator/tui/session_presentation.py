"""Bounded prepared transcript presentations, independent of owner attachment.

The owner and its canonical state stay authoritative. This module only owns
widgets and replay bookkeeping; preparing one has no subscription, prompt,
acknowledgement, or reference to the currently selected app session.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

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

#: The in-transcript seam between what the agent still sees and what it does
#: not. Both halves of that sentence have to be on screen: the rows it points
#: at are REAL history (so the reader is not being told anything was lost) and
#: they are outside the model's context (so the reader is not misled into
#: thinking the agent can still refer to them).
#:
#: The direction is ABOVE, and it is not interchangeable with "below". A
#: transcript paints oldest-at-top, so the pre-compaction rows sit above this
#: marker and the rows below it are the newer ones the model still sees.
#: Design review round 1 (D1) measured both halves of the original "older
#: messages below" wording against real mounted block positions and against
#: ``mode="context"`` membership: rows BELOW a marker were 100% still in
#: context — exactly what the sentence claimed the agent could not see — while
#: the rows above were 0-1%. Audit reading is the one task where someone is
#: reasoning about which rows the agent could have used, so a marker pointing
#: the wrong way is worse for a trusting reader than the silent nothing it
#: replaced.
#:
#: The wrap budget at 80 columns is 70 text columns, and the MECHANICAL check
#: is ``NoticeBlock.body_budget(76) == 70`` rather than hand arithmetic (design
#: review round 2, D6). The chain, measured on a block mounted in an 80x30
#: pilot rather than reasoned on paper: an 80-column terminal gives a 78-column
#: screen; ``scrollbar-gutter: stable`` on ``TranscriptView`` permanently
#: reserves one more column whether or not the bar is visible (77); the view's
#: own left padding takes one (76, which is the width the block is actually
#: painted); ``_build`` folds at ``max(width - 2, 12)`` (74); and the hanging
#: glyph field is ``GLYPH_COLS`` = ``SPINE_INDENT + 2`` = 4, not 2 (70).
#: Corroborated by rendered block height at that geometry: a 70-character
#: string is one row, a 71-character string is two.
#:
#: This string is 66, so it sits 4 columns inside the budget, and the unit
#: guard pins 66 rather than 70 — a copy change that wants the headroom has to
#: move that guard deliberately. Do NOT re-derive the budget from the older
#: "68 usable, so 66" arithmetic: it landed on a safe number through two
#: COMPENSATING errors — it subtracted neither the reserved scrollbar column
#: nor the fold clamp, and charged the glyph 2 where it costs 4 — so it cannot
#: be carried to any other width. Round 1's "under ~76", also derived on
#: paper, produced a 71-character string that still wrapped in the frame. That
#: is why a longer string is checked in a RENDERED frame rather than counted:
#: it orphans its last word on a second line at every compaction, 48 times in
#: the reference journal (D2). Capture one with
#: ``scripts/audit_history_shot.py <dir> marker 80x30``.
COMPACTION_MARKER_NOTICE = "context compacted — earlier history above the agent no longer sees"


def live_projection_call_ids(session: Any) -> set[str]:
    """Call ids a settled replay must SKIP because they are executing NOW.

    ``executing() - pending()``, and the subtraction is the load-bearing
    half. A turn parked at an approval gate is ALSO a streaming one, so both
    accessors answer with exactly the call the gate is holding; seeding a
    projection with the un-subtracted set would skip that call out of the
    replay, and the skip path (``_paint_skipped_live_tool_rows``) paints
    ``running`` without consulting the gate — bypassing the "waiting wins"
    rule ``_mark_pending_tool_rows`` exists to enforce. With the held call
    left out of the seed, the replay mounts its row and the pending scan
    marks it ``waiting``: the honest state for a call parked on the USER,
    not on a tool.

    One helper because three sites ask this question — the visible seed in
    ``_project_settled_rows``, the offscreen seed in the sidebar prepare
    path, and the fold's own fallback for a target that never seeded — and
    a divergence between any two reopens the gate bug on exactly one path.
    Lives here rather than on the app so the fold can ask it without a
    circular import.
    """
    executing = getattr(session, "executing_display_tool_ids", None)
    if not callable(executing):
        return set()
    live = cast(set[str], executing())
    pending = getattr(session, "pending_display_tool_ids", None)
    if callable(pending):
        live -= cast(set[str], pending())
    return live


class CompactionMarkerBlock(NoticeBlock):
    """The compaction seam, spaced apart from whatever it lands beside.

    A plain ``NoticeBlock`` would be right in every respect but one. In the
    audit state the head notice sits directly above this row, and the two share
    a ``SPACING_KIND`` of ``notice``, so the adaptive rule stacks them flush
    (see :func:`needs_gap_above`: same kind, previous is one row, no gap). They
    also share the ``·`` glyph and the ``note`` ink, and after the round-1 copy
    fix they open with nearly the same words — "earlier history above — scroll
    up to load" over "context compacted — earlier history above …". Design
    review round 1 (D3) flagged the pair as indistinguishable, and the D1 fix
    alone did not resolve it: verified in a rendered 80x30 frame, the two rows
    still read as one wrapped block, with nothing to say that the first is a
    CONTROL (focusable, clickable, `enter`-bound) and this one is inert.

    ``SPACING_AIRY`` is the mechanism the tool ledger already uses for exactly
    this — "each row is a separate thing, not a paragraph of one" — so the seam
    takes a blank row above itself rather than earning a second glyph or a
    second ink. The designer's own preference, and it keeps ``NOTICE_GLYPHS``'
    deliberate sharing of ``·`` between ``info`` and ``note`` intact.
    """

    SPACING_AIRY = True


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

    def __init__(self, text: str, *, fold_width: int = 0) -> None:
        super().__init__(text, "note", fold_width=fold_width)
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
    """The offer to put an undelivered draft back in the composer.

    Reached when the composer is NOT empty at the moment a send fails — the
    common case for a refusal, whose refit runs ~315 ms on a thread, long
    enough for the user to have started their next thought. The draft parks in
    ``source.unsent`` and this row is how it comes back.

    The label has three jobs the bare ``Restore unsent prompt`` did none of
    (design round 1, D5):

    **It says WHICH prompt.** One row reading the same words whatever it holds
    cannot be told from another, and the user has no way to know whether the
    thing on offer is the message they care about or carries their attachment.
    So the opening words of the draft and its attachment count ride the label.

    **It says it is INTERACTIVE.** The row is focusable with ``enter`` and
    click bindings, but nothing in the pixels said press or click, so the
    affordance existed only for a user who tried it. The key is named in the
    label, which is the same way the app tells the user about every other
    non-obvious key.

    **It is ``note``, not ``warning``.** It rendered in the same amber as the
    refusal directly beneath it, so two rows of identical ink said two
    different kinds of thing — a state to recover from, and a failure to act
    on. ``note`` is the tier ``NoticeBlock`` documents for "the answer to
    something the user just did", which is exactly what an offer to restore is.
    """

    BINDINGS = [Binding("enter", "restore", "Restore unsent prompt", show=False)]

    #: Characters of the draft quoted in the label. Long enough to tell two
    #: drafts apart, short enough that the row stays one line beside the
    #: attachment clause and the key hint at ordinary widths.
    _PREVIEW_CHARS = 32

    class Requested(Message):
        def __init__(self, notice: DraftRecoveryNotice) -> None:
            super().__init__()
            self.notice = notice

    def __init__(self, source_token: str, draft: SessionDraft) -> None:
        super().__init__(self._label_for(draft), "note")
        self.source_token = source_token
        self.draft = draft
        self.add_class("interactive-notice")

    @staticmethod
    def _label_for(draft: SessionDraft) -> str:
        """``↩ restore unsent prompt "…" (1 image) — enter``.

        The glyph is the one the row's action means (put this back), and it
        leads because the row is an OFFER rather than a report. Degrades
        cleanly: a draft with no text quotes nothing, one with no attachments
        says nothing about them, and the key hint is always present because it
        is the part the user cannot discover any other way.
        """
        from local_operator.tui.widgets.editor import ATTACHMENT_MARKER

        parts = ["↩ restore unsent prompt"]
        # One line of it, whitespace collapsed: a multi-line draft would
        # otherwise put its second line into this row's own wrap. Attachment
        # markers come out first — the restored draft carries an `[Image #1]`
        # citation for every attachment, and quoting them here would spend the
        # preview's whole budget restating what the count clause says next.
        preview = " ".join(ATTACHMENT_MARKER.sub("", draft.text).split())
        if preview:
            if len(preview) > DraftRecoveryNotice._PREVIEW_CHARS:
                preview = preview[: DraftRecoveryNotice._PREVIEW_CHARS - 1].rstrip() + "…"
            parts.append(f'"{preview}"')
        images = sum(
            1 for attachment in draft.attachments.values() if getattr(attachment, "image", None)
        )
        if images:
            parts.append(f"({images} image{'s' if images != 1 else ''})")
        return " ".join(parts) + " — enter"

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
    #: Call ids of the turn still executing on the session being projected,
    #: captured at projection time by the caller that knows the session (the
    #: app for the visible transcript, the prepare caller for an offscreen
    #: presentation). ``replay_tool_call`` skips the settled row for these —
    #: the live path owns that call's one visible row — and the empty default
    #: is the COLD-resume answer: no live turn, nothing to skip.
    _projection_live_call_ids: set[str] = field(default_factory=set)
    #: The CALL OBJECTS the projection skipped because their call id is live,
    #: in transcript order. Read by the projection's owner after the fold to
    #: paint the one visible row for a still-running call that no live path
    #: will paint (a local resume onto a turn already in flight). Cleared with
    #: the ids above.
    _projection_skipped_live: list[Any] = field(default_factory=list)


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
    _projection_live_call_ids: set[str]
    _projection_skipped_live: list[Any]

    def _transcript_view(self) -> TranscriptView: ...

    def _append_block(
        self, block: Any, *, ends_empty_state: bool = True, pin_tail: bool = False
    ) -> None: ...

    def _append_image_blocks(
        self,
        images: list[ImageContent],
        *,
        marker_text: str | None = None,
        fold_width: int = 0,
    ) -> list[ImageBlock]: ...

    def _painted_tool_card(self, call_id: str) -> Any: ...

    def _settle_painted_tool_card(self, card: Any, result: Any) -> None: ...

    def _replay_tool_call(
        self,
        call: Any,
        results: dict[str, Any],
        *,
        user_run: bool = False,
        fold_width: int = 0,
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
        self,
        images: list[ImageContent],
        *,
        marker_text: str | None = None,
        fold_width: int = 0,
    ) -> list[ImageBlock]:
        # ``fold_width`` is accepted to satisfy ``ReplayTarget``: a prepared
        # replay is built OFFSCREEN, so there is no laid-out destination to
        # name and the caller passes nothing (its blocks are folded when the
        # presentation is committed into a mounted view).
        return append_image_blocks(
            self,
            images,
            marker_text=marker_text,
            navigation_visible=False,
            fold_width=fold_width,
        )

    def _painted_tool_card(self, call_id: str) -> None:
        return None

    def _settle_painted_tool_card(self, card: Any, result: Any) -> None:
        raise AssertionError("a prepared replay cannot contain a live tool card")

    def _replay_tool_call(
        self,
        call: Any,
        results: dict[str, Any],
        *,
        user_run: bool = False,
        fold_width: int = 0,
    ) -> None:
        # Zero for a prepared replay — see `_append_image_blocks` above.
        replay_tool_call(self, call, results, user_run=user_run, fold_width=fold_width)

    def prepare(
        self,
        history: list[Any],
        *,
        bound: int = 12,
        anchor_id: str = "",
        live_call_ids: set[str] | None = None,
    ) -> None:
        # Snapshot the session's in-flight calls NOW, before the fold: the
        # answer can change mid-projection, and a half-guarded tail is the
        # duplicate this field exists to prevent.
        self._projection_live_call_ids = set(live_call_ids or ())
        self._projection_skipped_live = []
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

            # No width: a prepared replay is authored offscreen for a parked
            # view (see `project_settled_rows`' `fold_width` note), and this
            # notice is folded when that view is laid out.
            self._resume_head_notice = OlderHistoryNotice(RESUME_OLDER_NOTICE)
            self.blocks.insert(0, self._resume_head_notice)
        if self._resume_pending_tail:
            self._resume_tail_notice = HistoryPageNotice()
            self.blocks.append(self._resume_tail_notice)
        self._block_sink = None
        # One projection's liveness answer must not leak into the next pass:
        # the empty set is the cold-resume default, the only correct answer
        # when nobody re-seeds it.
        self._projection_live_call_ids = set()


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
    #: The `interrupted` row the app painted for this conversation's own turn,
    #: still waiting for the anchor its session publishes for that outcome.
    #:
    #: Carried for the same reason `held_steer_blocks` is: it names a widget in
    #: THIS presentation's transcript, so it belongs to the conversation rather
    #: than to the app. Left behind, a switch away and back let the attention
    #: poller append a second `Interrupted` under the live `interrupted` — the
    #: duplicate row `OperatorApp._adopt_own_interrupt_notice` exists to remove.
    own_interrupt_notice: Any = None
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
    self: ReplayTarget,
    history: list[Any],
    *,
    bound: int | None = None,
    end: int | None = None,
    fold_width: int = 0,
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

    ``fold_width`` is the width every block this pass BUILDS will be given.
    Zero means "not supplied", and it is only right where there is no
    destination to name: a prepared replay authors offscreen for a parked view
    and is laid out inside it before that view is revealed.

    Every other caller has a destination and must name it, because a block
    built without one folds at the 80-column fallback, pins that fold as its
    height, and is re-authored by the first layout's resize — a second build
    per block and a painted frame whose rows wrap at 78 cells inside a 96 or
    146-cell pane. `OperatorApp._project_settled_rows` derives it from the live
    transcript for the tail, the reconnect gap replay, the sidebar commit's
    top-up and the older-page collect, so those four cannot disagree about the
    destination. It is threaded to CONSTRUCTION rather than applied after,
    because a block that wraps in ``__init__`` never reads a hint set later.

    The one authoring seam on a live transcript that still has no width is
    ``TranscriptView.append_block`` / ``OperatorApp._append_block``: a live
    prompt, notice or running card is built first and mounted second, so its
    rows come out of the fallback too and are saved only by that same resize.
    Deliberately not part of this seam: no painted narrow frame was observed on
    it (the append's mount and resize both complete before the paint; measured
    at 150x40 and 160x40 by review round 1 and the design round), so it is a
    wasted build rather than a visible defect — recorded in the PR's "not
    addressed" list, and it cannot be fixed from here because a hint applied at
    append arrives after the rows exist.
    """
    from contextlib import nullcontext

    from local_operator.compaction.marker import (
        COMPACTION_MARKER_TYPE,
        COMPACTION_REFUSED_TYPE,
    )
    from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE

    # The row DECISIONS this fold shares with the phone's. Held outside both
    # hosts so neither owns them: every divergence the convergence review
    # found was a decision one surface made and the other missed
    # (docs/design/history-fold-convergence.md §3).
    from local_operator.harness.rows import (
        assistant_row_text,
        assistant_stop_notice,
        compaction_refused_notice,
        gate_timeout_notice,
        is_harness_chrome,
        user_row_text,
    )
    from local_operator.tui.app import (
        PEER_MESSAGE_MESSAGE_TYPE,
        RESUME_OLDER_NOTICE,
        WAKE_PROMPT_MESSAGE_TYPE,
        AssistantBlock,
        PeerMessageBlock,
        UserBlock,
        WakeBlock,
        _resume_tail_start,
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
        # The width goes to construction here for the same reason every other
        # block in this pass gets it: a notice wraps itself in `__init__`, and
        # this one is authored and mounted in the same pass as the rows below
        # it, so without it the row is folded at the 80-column fallback before
        # any hint can reach it (QA round 1, Q1 caught exactly this row:
        # `OlderHistoryNotice box=96 authored at 80`).
        #
        # The pane where that is VISIBLE is a narrow one, not the 96-cell pane
        # the `box=` readout above happens to carry (design round 2, D6): the
        # sentence is 36 cells and fits on one row at every pane above ~44, so
        # at 100x30 and 60x20 this notice paints identically before and after.
        # Measured at 40x20 (pane 36): the fallback build is one row ending
        # `…scroll up`, and this one wraps to a second, hanging-indented row —
        # `…scroll` / `up to load`. Below the fallback the width is the
        # difference between the whole sentence and a truncated one.
        notice = OlderHistoryNotice(RESUME_OLDER_NOTICE, fold_width=fold_width)
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
                        self._append_block(
                            WakeBlock(
                                str(details.get("text", "")),
                                catchup=False,
                                fold_width=fold_width,
                            )
                        )
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
                            fold_width=fold_width,
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
                self._append_block(
                    NoticeBlock(gate_timeout_notice(details), kind="warning", fold_width=fold_width)
                )
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
                text, kind = compaction_refused_notice(details)
                self._append_block(NoticeBlock(text, kind=kind, fold_width=fold_width))
                appended = True
                continue
            # The compaction boundary itself. The replay layer has always
            # emitted this row, and it rendered as NOTHING: it is a custom
            # message, so it fell past every branch above and then past the
            # role-based handling below, which drops what it does not
            # recognise. That was survivable while the row only ever sat at the
            # very top of the model's replay; it is not survivable now that
            # audit paging puts one at each compaction MID-transcript, because
            # the reader would scroll from live conversation into
            # pre-compaction history with no sign of the seam.
            #
            # `note`, not `info`, for the reason `RESUME_UNREACHABLE_NOTICE`
            # is: this answers "where did my history go", and `info` maps to
            # `dim`, which measures below the AA contrast floor on the light
            # theme. Nothing went wrong here, so it is neither a warning nor an
            # error — the rows below are real history, they are simply outside
            # what the agent can still see.
            if getattr(message, "custom_type", None) == COMPACTION_MARKER_TYPE:
                self._append_block(
                    CompactionMarkerBlock(
                        COMPACTION_MARKER_NOTICE, kind="note", fold_width=fold_width
                    )
                )
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
                #
                # The list itself lives in ``harness/rows.py`` so the phone
                # fold reads the SAME three. It previously kept its own
                # partial copy — suppressing the connectivity prompt while
                # painting the other two as the user's own words.
                if is_harness_chrome(text):
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
                text = user_row_text(text)
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
                    self._append_block(UserBlock(text, len(replay_images), fold_width=fold_width))
                    self._append_image_blocks(
                        replay_images, marker_text=text, fold_width=fold_width
                    )
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
            # Through the shared helper even though this loop already
            # stripped: the DECISION about what an assistant row shows is the
            # thing both surfaces must read from one place. Leaving it as a
            # bare truthiness test here is what let the phone's own bare test
            # drift — the helper is only load-bearing if both hosts call it.
            if assistant_row_text(text):
                block = AssistantBlock()
                block.completion_anchor_id = str(getattr(message, "id", ""))
                # Before `update_text`: this block authors its rows through the
                # fold ladder on every update, and a hint set afterwards would
                # only reach a rebuild that has already happened.
                block.set_fold_hint(fold_width)
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
                self._replay_tool_call(call, results, user_run=user_run, fold_width=fold_width)
                appended = True
            # A refused, failed or interrupted turn needs a notice the prose
            # alone does not carry — a refusal fires even when the model
            # streamed some prose first, while error/aborted fire only for a
            # turn that produced nothing at all. The decision is shared with
            # the phone fold, which had NO stop_reason branch and therefore
            # showed a truncated answer as complete and a failed turn as
            # silence (§3, D2-D4).
            notice = assistant_stop_notice(
                text=text,
                has_tool_calls=bool(tool_calls),
                stop_reason=getattr(message, "stop_reason", None),
                provider_payload=getattr(message, "provider_payload", None),
            )
            if notice is not None:
                reason, severity = notice
                self._append_block(NoticeBlock(reason, severity, fold_width=fold_width))
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
    self: ReplayTarget,
    call: Any,
    results: dict[str, Any],
    *,
    user_run: bool = False,
    fold_width: int = 0,
) -> None:
    """Mount one settled tool row for a call from a previous session.

    The card is built exactly as a live one is — same constructor, same
    summary derivation from the arguments — so a resumed row is
    indistinguishable from the row the user watched run, including its
    duration: the harness persists the executor's measured interval as
    ``provider_payload.duration_s`` beside the result's ``details``, and it is
    restored here rather than recomputed from when this row was mounted.

    One call has exactly ONE visible row. When the transcript carries an
    outcome, a card the live path already painted is settled in place rather
    than doubled by a second row. When the call is live IN THIS PROCESS right
    now — the replay ran against a session whose turn is still in flight, as
    ``/resume`` onto a running conversation does — mounting a settled row at
    all is the reported duplicate: the running turn's own ``ToolStarted``
    paints (or has painted) the live row, and this second card would sit
    beside it, overcount "running N tools", and stamp ``⊘ interrupted`` on a
    call that has not stopped. Skipping the settled row loses nothing: the
    live row owns the call's intent (which the transcript does not persist)
    and its real start time. Where no live row exists yet — the local adopt,
    whose ``ToolStarted`` predates this process's subscription — the
    projection's owner paints the one row afterwards
    (``_paint_skipped_live_tool_rows``) and registers it with the event
    controller, so it settles through the ordinary ``on_tool_ended`` path.
    A COLD resume has no live turn, so ``executing_display_tool_ids()`` is
    empty there and killed-mid-turn calls still render ``interrupted``
    exactly as before.
    """
    from local_operator.tui.app import ImageContent, ToolCard, _first_line
    from local_operator.tui.widgets.tool_card import parse_duration

    call_id = getattr(call, "id", "") or ""
    result = results.get(call_id)
    if result is not None:
        # The outcome is recorded, so the call cannot still be running: a card
        # the live path already painted (a resumed live projection, a
        # reconnect's pre-disconnect card) is settled in place through the same
        # derivation instead of being doubled by a fresh row — review round 4,
        # MINOR-1 built exactly this pairing for the gap replay.
        painted = self._painted_tool_card(call_id)
        if painted is not None:
            self._settle_painted_tool_card(painted, result)
            return
    else:
        # The snapshot the caller seeded for this projection first; a
        # ReplayTarget carrying the session itself (the app) falls back to
        # asking the session directly through the SAME subtracted question
        # the seed uses, so a caller that never seeded cannot reopen the
        # gate-parked-skip bug (a target with no session — a prepared
        # presentation — answers nothing and mounts, which for a prepared
        # tail the commit path repaints is the pre-existing behaviour).
        live_ids: set[str] = self._projection_live_call_ids
        if not live_ids:
            session = getattr(self, "_session", None)
            live_ids = live_projection_call_ids(session)
        if call_id in live_ids:
            # Record it, so the projection's owner can paint the ONE row for
            # this call when no live path is going to (a local resume: the
            # turn's ToolStarted fired before this process/subscription
            # existed, so no card will arrive for it). The replay itself still
            # mounts nothing — the owner paints it after the fold, where it
            # knows whether a card is already on screen.
            self._projection_skipped_live.append(call)
            return
        # The gate-free seed answered "not live", but the seed is a snapshot:
        # a call parked at an approval gate is subtracted out of it (so the
        # row paints `waiting`, not `running`) while the live ToolStarted
        # already mounted a card during adoption. Mounting a second row here
        # is the same duplicate the live-skip above prevents, one arm over —
        # consult the already-painted registry exactly as the outcome branch
        # does (QA round 2, Q-R2-1).
        if self._painted_tool_card(call_id) is not None:
            return
    card = ToolCard(
        call_id,
        getattr(call, "name", "") or "",
        getattr(call, "arguments", None) or {},
        user_run=user_run,
    )
    # Before the `restore`/`mark_*` calls below re-author the row: the
    # constructor's own build is the throwaway one (a detached card has no
    # width to ask and falls to the console fallback), and every state call
    # that follows rebuilds through the fold ladder — so this is the moment
    # the card can be told the width its rows should be authored at.
    card.set_fold_hint(fold_width)
    self._append_block(card)
    if result is None:
        # No result recorded: the session ended between the call and its
        # answer. Showing it as complete would invent an outcome, and there is
        # no measured interval to restore either — the distinction that governs
        # the duration column is result-present vs result-absent, and this is
        # the only arm on the absent side.
        card.restore(state="interrupted")
        return
    result_text = getattr(result, "text", "") or ""
    payload = getattr(result, "provider_payload", None) or {}
    is_dict = isinstance(payload, dict)
    details = payload.get("details") if is_dict else None
    # Validated, never trusted: the key is absent on every row written before
    # durations were persisted, and a transcript is an on-disk file another
    # process may have written. `parse_duration` degrades anything that is not
    # a finite non-negative number to ``None``, which paints the same blank
    # column a legacy row paints — the one honest answer when the interval is
    # unknown. Replay must not fail on a bad value, and must not invent a
    # ``0.0s`` that says the tool returned instantly.
    duration_s = parse_duration(payload.get("duration_s")) if is_dict else None
    if getattr(result, "is_error", False) and result_text.startswith("aborted ("):
        # An aborted call persists as an error result (the model-facing
        # shape), but the LIVE frame it came from was the dim shut
        # `interrupted ⊘` row. Replaying it through the error branch would
        # reopen the user's own Esc as a red failure (design round 1, D1).
        #
        # The parenthesised prefix identifies ONE producer, and it is not the
        # agent loop: `execute_bash` builds this text itself via `_error(...)`
        # (`tools/builtin.py:1478`, `:1990`), as does `tools/eval.py:882`.
        # `harness/loop.py`'s synthetic abort is `ABORTED_RESULT_TEXT =
        # "aborted"` with NO parenthesis, so every result the loop parks a
        # duration onto fails this guard and takes the plain error arm below.
        #
        # So `duration_s` is passed for faithfulness, not for a population we
        # can point at today. `_error(...)` sets no `duration_s`, and the only
        # producer that MEASURES one is the agent loop (`loop.py::park`, which
        # stamps `result.duration_s` from its own `time.monotonic()` span)
        # — review round 2 swept 37 real aborted runs (model-issued bash,
        # parallel-batch races, eval kernel aborts) and produced the
        # `aborted (` + `duration_s` conjunction zero times. The arm is
        # therefore inert but correct: `park()` stamps any NORMALLY returned
        # result, so the moment a producer returns this text with a measured
        # interval the row shows it instead of silently dropping it.
        # `tools/eval.py:1004` (`aborted (…): kernel killed mid-run`) is the
        # plausible future one — it returns normally rather than by
        # cancellation — but neither reviewer could drive a turn into that
        # branch or prove it unreachable, so its status is unsettled.
        #
        # Not covered here: a bang-mode `! cmd` the user stopped. That row
        # persists through `session/shell_record.py` →
        # `Message.tool_result`, which DOES carry a `provider_payload` when
        # the result has one (a spilled capture writes `details['spill']`).
        # It still replays blank, for a different reason: nothing measures a
        # bang command, so `duration_s` is `None`. The terminal runs it
        # outside a turn, so the loop's `park()` — the only caller that
        # stamps an interval — never sees it, and `execute_bash` reports no
        # duration of its own. A successful `! echo hi` replays blank for the
        # same reason. The producer gap is upstream and out of scope.
        card.restore(state="interrupted", duration_s=duration_s)
        return
    if getattr(result, "is_error", False):
        card.restore(
            state="error",
            result_text=result_text,
            details=details,
            error=_first_line(result_text),
            duration_s=duration_s,
        )
    else:
        card.restore(
            state="success",
            result_text=result_text,
            details=details,
            duration_s=duration_s,
        )
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
    fold_width: int = 0,
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
                fold_width=fold_width,
            )
        except Exception:
            logger.debug("image block construction failed", exc_info=True)
            continue
        self._append_block(block)
        mounted.append(block)
    return mounted
