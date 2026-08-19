"""Session public API contract.

Implemented by ``local_operator.session.session.Session`` (stream A) and
programmed against by the TUI (stream D), exec mode (stream E), and the
server facade (integration). Keeping the surface as a Protocol lets the UI
and headless modes build and test against a fake session before the real one
lands.

Event delivery semantics: ``subscribe`` handlers receive
``AgentEvent`` instances in emission order; a handler may be sync or async.
``agent_end`` may arrive AFTER a subsequent ``agent_start`` when a turn was
superseded — UIs must handle that (see docs/REWRITE.md, stream D).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from local_operator.harness.approval import ApprovalGate
from local_operator.harness.types import (
    AgentMessage,
    AskUserFn,
    EventHandler,
    ImageContent,
    Message,
    ModelSpec,
    Usage,
)
from local_operator.session.naming import ConversationName


@dataclass(frozen=True, slots=True)
class CompactionOutcome:
    """What one explicit compaction request did — see :meth:`SessionProtocol.compact_now`.

    A manual trigger can be pressed in states the automatic gate never sees (a
    turn still streaming, a context too small to be worth a pass, a context
    already compacted), so "did not run" is a first-class answer that has to
    carry WHY: a host that cannot tell a refusal from a no-op reproduces the
    bug where ``/compact`` silently changed nothing.

    ``reason`` is the stable code (``already_running``, ``turn_running``,
    ``disabled``, ``nothing_to_compact``, ``cut_not_replayable``,
    ``below_threshold``, ``unavailable``, ``failed``); ``detail`` is the
    one-sentence explanation a front end can show verbatim, written HERE rather
    than in each host so the TUI, exec mode and the server cannot each invent
    their own wording for the same refusal.

    ``tokens_before`` is the figure the compaction gate acted on —
    ``max(provider-reported context, local estimate)`` — so the receipt agrees
    with the status band the user was just looking at. ``tokens_after`` is
    ``tokens_before`` minus the pass's saving, where the saving is the
    HISTORY-only difference measured by one local ruler on both sides (archive
    frames re-priced at the provider's own image billing). Subtracting a
    history-only after-figure from a full-request before-figure would count
    the system blocks and tool schemas — which a compaction does not touch —
    as if the pass had removed them; keeping the overhead on both sides is
    what lets a host subtract the pair from its own reading safely.
    """

    ran: bool
    reason: str = ""
    detail: str = ""
    strategy: str = ""
    tokens_before: int = 0
    tokens_after: int = 0


@runtime_checkable
class SessionProtocol(Protocol):
    """The one object every front end talks to."""

    # --- identity / state -------------------------------------------------
    @property
    def session_id(self) -> str: ...

    @property
    def agent_id(self) -> str: ...

    @property
    def is_streaming(self) -> bool: ...

    @property
    def model_label(self) -> str:
        """Human-readable ``provider/model`` for status lines."""
        ...

    @property
    def model(self) -> ModelSpec:
        """The active spec (provider/model_id/base_url/context_window)."""
        ...

    def set_model(self, model: ModelSpec) -> None:
        """Swap the model spec; in force from the very next provider call.

        The TUI's ``/model <provider>/<id>`` path calls this after building a
        new spec, so no session teardown is required. Also changes compaction
        thresholds for the new context window.

        Not "from the next turn": an implementation is expected to reach the
        RUNNING turn too. A turn is a chain of provider calls with tool batches
        between them, and a user switching model mid-turn is doing it because
        the running model is doing badly, so the switch lands at the next call
        boundary. Whatever is already in flight finishes on the spec it was
        issued with \u2014 a switch must never split one response across two models.
        :class:`~local_operator.session.session.Session` implements this by
        handing the loop a ``LoopConfig.get_model`` resolver.
        """
        ...

    @property
    def goal(self) -> str:
        """The session's standing objective ("" when unset)."""
        ...

    def set_goal(self, text: str) -> str:
        """Set or clear the standing objective; returns what was stored.

        Backs ``/goal``. The objective rides the system prompt's volatile
        tail, so it applies from the next turn onward.
        """
        ...

    @property
    def conversation_name(self) -> str:
        """The conversation's title ("" until one is set or generated)."""
        ...

    @property
    def conversation_name_state(self) -> ConversationName:
        """The title holder, not just the string.

        A host that re-titles a drifting conversation needs the ``user_set``
        precedence flag before it spends a call: an explicit rename outranks
        every generated title forever, and reading only the text cannot tell
        a human's name from a model's.
        """
        ...

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        """Name the conversation; returns the title in force afterwards.

        ``user_set=True`` marks an explicit rename, which a later
        auto-generated title must not overwrite.
        """
        ...

    async def complete_once(self, system: str, prompt: str) -> str:
        """One cheap, isolated, single-attempt provider call for a host errand.

        Not a turn: no tools, no history, no transcript entry. Hosts use it
        for small derived text (conversation auto-naming) without rebuilding
        the provider's auth cascade. It runs CONCURRENTLY with a live turn, so
        an implementation must make it unable to move anything the turn
        depends on — see ``ChatRequest.isolated``.
        """
        ...

    def history(self) -> list[AgentMessage]:
        """The conversation as replayed into LLM context.

        Read-only for RENDERING (a resumed session's transcript back on
        screen): returns the messages the loop sees, in order — user prompts,
        assistant replies, tool results. A front end mounts them as blocks;
        it must NOT mutate them. Empty before the first prompt on a fresh
        session; on ``--resume`` it carries the prior conversation.
        """
        ...

    async def complete_aside(
        self,
        turns: list[AgentMessage],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> str:
        """Answer a side question against the live context WITHOUT joining it.

        Reads what a real turn reads — the live system blocks and the whole
        message list — and writes nothing: no transcript entry, no append to
        the conversation, no events. ``turns`` are appended for this request
        only, which is how a caller supplies the side question itself (and any
        in-flight assistant text it is painting but the context does not carry
        yet). No tools.

        It is NOT free: the request carries the whole conversation. Nothing is
        recorded, so ``on_usage`` reports the provider's own figures to
        whatever the host counts spend with.

        Backs the TUI's ``/btw`` aside overlay. The no-trace guarantee is the
        feature: dismissing the overlay must leave the conversation, and the
        model's view of it, exactly as they were found.
        """
        ...

    async def adopt_aside(self, messages: list[Message]) -> None:
        """Promote an off-the-record aside exchange into the conversation.

        The user's explicit opt-out of :meth:`complete_aside`'s no-trace
        contract: appends the messages as ordinary turns to both the live
        context and the transcript. Raises while a turn is running — the loop
        owns the message list for the duration, and splicing into a tool batch
        makes it unsendable.
        """
        ...

    # --- context ----------------------------------------------------------
    async def compact_now(self) -> CompactionOutcome:
        """Compact the conversation context NOW, on the user's request.

        THE SAME PASS the automatic gate runs when the context fills up — same
        strategy resolution (snapcompact for a vision model, a language summary
        otherwise), same cut point, same transcript entry, same
        ``compaction_start``/``compaction_end`` events — with the threshold
        check skipped, because the user asking IS the trigger. Backs the TUI's
        ``/compact``.

        Never raises for a state it can describe: a turn still running, a
        context too small to be worth summarizing, compaction disabled in
        config. Those come back as a :class:`CompactionOutcome` with
        ``ran=False`` and a reason to show, so a host can always say why
        nothing happened.
        """
        ...

    # --- driving turns ----------------------------------------------------
    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        """Run one user turn to completion (awaitable) or raise.

        ``images`` are attachments pasted into the prompt; they ride the same
        message as the text so the model reads them as one turn.
        """
        ...

    async def seed_history(self, messages: list[Message]) -> None:
        """Prime the conversation from a host-supplied history.

        Once-only and pre-prompt: a no-op once the context carries messages
        (transcript replay populated them) or after the first turn. The server
        facade needs it for the two paths where the transcript is not the
        history source — stateless chat and non-persisted agent chat — so the
        provider sees the same history the response envelope echoes.
        """
        ...

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        """Inject a steering message into the running turn (interrupts tool
        batches at the next boundary)."""
        ...

    def abort(self, reason: str = "interrupted") -> None:
        """Abort the running turn; the engine emits an aborted agent_end."""
        ...

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """Cancel every running subagent; returns how many were stopped.

        Separate from :meth:`abort`, which stops only THIS session's turn. A
        subagent is a child session with its own turn and its own spend, so a
        stopped parent does not stop it. Backgrounded ``bash`` jobs are not
        touched — ``background=true`` exists to outlive the turn.

        The count is part of the contract: a host prints it, and "nothing was
        running" has to be distinguishable from "children were stopped".
        """
        ...

    def running_subagents(self) -> int:
        """How many subagents :meth:`cancel_subagents` would stop right now.

        The counterpart to that call, so a host can OFFER the stop ("N still
        running — press again") with the same number the stop will report. The
        two must come from one predicate or the confirmation can contradict the
        offer the user just acted on.
        """
        ...

    def set_approval_handler(self, handler: ApprovalGate | None) -> None:
        """Replace the host's tool-approval gate for write/exec tier tools.

        A front end that OWNS the terminal must own approvals with it: the
        default gate reads a y/N answer off stdin, which a full-screen UI has
        taken over, so leaving it installed hangs the turn instead of asking
        anyone. The handler is read when the per-turn tool context is built, so
        installing one mid-session applies from the next tool call. ``None``
        restores auto-approval (what ``--yolo`` already does).
        """
        ...

    def set_ask_handler(self, handler: AskUserFn | None) -> None:
        """Install the surface that puts the ``ask`` tool's questions to the user.

        Declared beside the approval gate because it is the same kind of hook —
        a front end that owns the terminal is the only thing that can draw a
        picker — and because installing it is what makes the ``ask`` tool exist
        at all: its createIf builder gates on the hook, so a host that never
        calls this (a server, exec mode, or any subagent) advertises no question
        it could only block on. Read when the per-turn tool context is built, so
        installing one mid-session applies from the next tool call.
        """
        ...

    # --- events -----------------------------------------------------------
    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register an event handler; returns an unsubscribe callable."""
        ...

    # --- lifecycle --------------------------------------------------------
    async def dispose(self) -> None: ...
