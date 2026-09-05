"""Bounded prepared transcript presentations, independent of owner attachment.

The owner and its canonical state stay authoritative. This module only owns
widgets and replay bookkeeping; preparing one has no subscription, prompt,
acknowledgement, or reference to the currently selected app session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from textual import events
from textual.binding import Binding
from textual.message import Message

from local_operator.harness.types import ImageContent
from local_operator.tui.widgets.image_block import ImageBlock
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptBlock,
    TranscriptView,
)


class HistoryPageNotice(NoticeBlock, can_focus=True):
    BINDINGS = [Binding("enter", "more", "More recent messages", show=False)]

    class Requested(Message):
        def __init__(self, notice: HistoryPageNotice) -> None:
            super().__init__()
            self.notice = notice

    def __init__(self) -> None:
        super().__init__("More recent messages below", "note")

    def action_more(self) -> None:
        self.post_message(self.Requested(self))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_more()


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
        block.navigation_anchor_id = self._projection_message_id
        block.navigation_anchor_part = self._projection_part
        self._projection_part += 1
        self.blocks.append(block)

    def _append_image_blocks(
        self, images: list[ImageContent], *, marker_text: str | None = None
    ) -> list[ImageBlock]:
        return append_image_blocks(self, images, marker_text=marker_text)

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
        anchor = next((index for index, message in enumerate(history) if str(getattr(message, "id", "")) == anchor_id), None) if anchor_id else None
        end = min(len(history), anchor + bound) if anchor is not None else None
        project_settled_rows(self, history, bound=bound, end=end)
        if self._resume_pending_head:
            from local_operator.tui.app import RESUME_OLDER_NOTICE

            self._resume_head_notice = NoticeBlock(RESUME_OLDER_NOTICE, "note")
            self.blocks.insert(0, self._resume_head_notice)
        if self._resume_pending_tail:
            self._resume_tail_notice = HistoryPageNotice()
            self.blocks.append(self._resume_tail_notice)
        self._block_sink = None


@dataclass
class SessionPresentation:
    replay: PreparedReplay
    revision: int = 0
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
        notice = NoticeBlock(RESUME_OLDER_NOTICE, "note")
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
                # Replayed completions must stay acknowledgeable: the viewed
                # receipt is keyed by the anchoring message id, so a block
                # rebuilt from history carries the same id the live render
                # would have (upstream #697).
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
    self: ReplayTarget, images: list[ImageContent], *, marker_text: str | None = None
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
            block = ImageBlock(image.data or None, image.mime_type, label=label)
        except Exception:
            logger.debug("image block construction failed", exc_info=True)
            continue
        self._append_block(block)
        mounted.append(block)
    return mounted
