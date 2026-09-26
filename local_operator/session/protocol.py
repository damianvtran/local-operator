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

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

from local_operator.harness.approval import ApprovalGate
from local_operator.harness.types import (
    AgentMessage,
    AskUserFn,
    EventHandler,
    ImageContent,
    Message,
    ModelSpec,
    ToolResult,
    Usage,
)
from local_operator.session.naming import ConversationName
from local_operator.session.spend import SessionSpend

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Deferred: ``frontend_state`` imports ``tui.costs``, whose package
    # ``__init__`` imports THIS module, so a runtime import here is a cycle.
    # The annotation is all that is needed — ``from __future__ import
    # annotations`` above makes every annotation in this file a string, and a
    # Protocol member's type is never evaluated at runtime.
    from local_operator.session.frontend_state import FrontendSessionState

    # Same shape, same reason: ``mark_goal_done`` returns the settled entry, and
    # naming the real type keeps the caller's ``entry`` from degrading to ``Any``
    # (``tui/app.py`` passes it straight into ``goal_done_answer``). ``GoalState``
    # too: the judge's own record is what ``GoalRecordProtocol.goal_judge_state``
    # hands the driver beside the wire shape.
    from local_operator.session.goal import GoalHistoryEntry, GoalJudgeState


def unanswered_tail_call_ids(messages: Sequence[Any]) -> set[str]:
    """Calls in the CURRENT turn's latest group that have no result yet.

    The one scan behind both "this row is not finished" display questions
    (``pending_display_tool_ids`` and ``executing_display_tool_ids``), which
    differ only in what makes the call unfinished — a gate parked in front of
    it, or the tool still executing. Held here rather than on either session
    class because the rule is a property of the MESSAGE TAIL, not of the
    transport: the local owner reads its in-memory context and a remote
    viewer its display window, and the two must answer identically for the
    same conversation or the same resumed row settles differently depending
    on which surface replayed it.

    The latest call group after the latest user boundary is the only eligible
    group, so old interrupted turns keep their ``⊘`` and are never revived by
    a later turn's liveness.
    """
    answered: set[str] = set()
    for message in reversed(messages):
        role = getattr(message, "role", "")
        if role == "user":
            break
        if role == "tool":
            answered.add(str(getattr(message, "tool_call_id", "")))
        calls = getattr(message, "tool_calls", None)
        if role == "assistant" and calls:
            return {call.id for call in calls} - answered
    return set()


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


#: Where the runtime that executes this session's turns lives, relative to the
#: process asking. ``"this-process"`` means the loop runs on this thread pool
#: and the transcript is written here; ``"this-machine"`` means a separate
#: process on this host (the ordinary `lop` viewer, reached over a loopback
#: socket); ``"another-machine"`` means the owner is a runtime on another device,
#: reached through this device's mesh relay; ``"unknown"`` is the conservative
#: arm for a facade that cannot prove any of them — it exists because the honest
#: answer to "is a config write going to govern this runtime?" is sometimes
#: "cannot tell", and collapsing that into a bool is what made the predicate it
#: replaces wrong five times.
#:
#: ``"another-machine"`` IS A REAL MEMBER NOW, and the reason the union was
#: three-valued is the reason it is four rather than three-with-a-lie. The
#: paragraph that used to stand here said a cross-host runtime "cannot occur"
#: because every listener and dialer in this tree binds or dials ``127.0.0.1``.
#: That is still true of every listener — the mesh relay's peer listener is the
#: one deliberate exception, and it is authenticated end to end — but it is no
#: longer true of the question this type answers: a ``RemoteOwner``
#: (``network/projection.py``) attaches a viewer here to a runtime there, over
#: exactly those loopback sockets chained through the relay. The answer in that
#: case is known, not merely safe, which is strictly more information than
#: ``"unknown"``: a config write here will NOT govern that runtime.
#:
#: Every ``==`` comparison over this union is therefore load-bearing, and
#: ``tests/unit/session/test_runtime_locality_union.py`` walks the tree for them.
RuntimeLocality = Literal["this-process", "this-machine", "another-machine", "unknown"]

#: Where an accepted-but-undelivered gate reply goes, for a host with a surface
#: for it: the gate's KIND and its IDENTITY, ``(kind, request_id,
#: question_index)`` — the tuple the viewer ladder keys bridges on (see
#: :meth:`ViewerSessionProtocol.set_gate_undelivered_handler`). Named rather than
#: spelled out because the identity half is what lets a host retract the receipt
#: THAT card wrote, and an unnamed tuple in a callback type is how that half gets
#: dropped from a signature by the next person.
GateUndeliveredHandler = Callable[[str, tuple[str, str, int] | None], None]


@runtime_checkable
class SessionProtocol(Protocol):
    """The one object every front end talks to."""

    # --- runtime role -----------------------------------------------------
    # Three predicates that name what hosts actually need to know about the
    # relationship between this process and the session's runtime. They exist
    # because an undeclared ``is_remote`` attribute conflated them: it was
    # written in exactly one place (``attached.py``), was never declared on this
    # protocol, and was read through ``getattr(session, "is_remote", False)`` at
    # every call site, where the default silently meant "in-process" and a typo
    # in the string was invisible to pyright.
    #
    # Since 0.46.0 `lop` builds a viewer for every local user, so that flag was
    # constant-True for every TUI session and the transport question it named
    # was dead. The same conflation was fixed site-locally four times
    # (#576, #609, #624, #625) and guarded a fifth
    # (``tests/unit/tui/test_noop_consumers.py``). A predicate that has been
    # wrong five times is a missing type, not a naming problem — so the three
    # surviving questions are declared here, separately, and checked.
    #
    # The flag itself is GONE. Every host now asks one of these three by name,
    # and ``tests/unit/session/test_viewer_protocol.py`` fails if a new
    # undeclared session flag of that shape is introduced.

    @property
    def owns_runtime(self) -> bool:
        """Whether THIS process runs the turn loop and writes the transcript.

        True on the runtime (`lop exec`, the server host, a subagent host),
        False on a facade attached to a runtime that lives elsewhere. It is the
        lifecycle question: may this process end the session in-process, does
        an in-process abort reach the loop, is auto-naming this process's job.

        Not the same as :attr:`outcome_is_synchronous` even though both are
        False for every `lop` TUI session today — a runtime could in
        principle admit a turn without running it to completion, and the two answers
        would then diverge. Keeping them separate is the point: the sites that
        ask them are asking different things.
        """
        ...

    @property
    def outcome_is_synchronous(self) -> bool:
        """Whether :meth:`prompt` returns only after the turn's terminal outcome.

        True where ``prompt()`` returns after the pipeline's ``finally`` has
        flushed ``agent_end``, so the caller HOLDS the outcome and may mark the
        turn failed or succeeded from its own stack. False where it returns on
        the runtime's durable-admission ACK — mid-turn, before the outcome exists
        — so the caller must wait for the event stream instead and must not
        infer success from the call returning.

        A host that gets this wrong paints a turn's result from a return value
        that carries no result. It is the meaning behind the ``outcome_known``
        and ``knows_outcome`` locals in the TUI.
        """
        ...

    @property
    def runtime_locality(self) -> RuntimeLocality:
        """Where the runtime executing this session's turns lives.

        The question a host asks before writing this machine's ``config.yml``
        and expecting the runtime to obey it: ``"this-process"`` and
        ``"this-machine"`` both mean the write governs the runtime (two
        terminals on one host share one config file), ``"unknown"`` means it
        cannot be proven and the caller should be conservative.

        Deliberately three-valued rather than boolean — see
        :data:`RuntimeLocality`.
        """
        ...

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
        """The SELECTED spec (provider/model_id/base_url/context_window)."""
        ...

    @property
    def effective_model(self) -> ModelSpec:
        """The spec ACTUALLY serving requests.

        Equals ``model`` except while a provider fallback is pinned, when it is
        the fallback's own spec. Front ends paint their model display from THIS
        — a display reading ``model`` during a fallback names a model that is
        not answering.
        """
        ...

    @property
    def effective_model_label(self) -> str:
        """``provider/model`` of the spec actually serving requests."""
        ...

    def set_model(self, model: ModelSpec, *, explicit: bool = False) -> None:
        """Swap the model spec; in force from the very next provider call.

        The TUI's ``/model <provider>/<id>`` path calls this after building a
        new spec, so no session teardown is required. Also changes compaction
        thresholds for the new context window.

        ``explicit`` marks a deliberate model CHOICE (``/model``, the phone's
        model switch) rather than an internal knob adjustment (``/effort``,
        sampling overrides). While a provider fallback is pinned, an explicit
        choice withdraws the pin even when it re-selects the model the fallback
        displaced — the user reclaiming their model ends the rescue route.

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
        tail, so it applies from the next model step (or the next turn when
        the session is idle).
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
        """One cheap, isolated, near-single-attempt provider call for a host errand.

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
        aside_instruction: bool = True,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> str:
        """Answer a side question against the live context WITHOUT joining it.

        ``aside_instruction`` says whether ``turns`` still need the off-record
        instruction applied. True (the default) is for a caller sending the RAW
        question — the seams (``ServingSessionHandle``, ``TuiSessionHandle``)
        wrap the last user turn, which is how every remote surface gets the
        instruction. False is the caller's declaration that it supplied its OWN
        instruction and the seam must not add a second: the TUI's ``/btw``
        overlay wraps ``ASIDE_PROMPT`` itself, and its goal-loop judge sends
        ``LOOP_JUDGE_PROMPT``, which is not an aside at all. IMPLEMENTATIONS
        THAT DO NOT WRAP accept the flag and ignore it (``Session.complete_aside``
        is the bare primitive), so a caller can pass it at every hop without
        knowing which hop it holds.

        Reads what a real turn reads — the live system blocks and the whole
        message list — and writes nothing: no transcript entry, no append to
        the conversation, no events. ``turns`` are appended for this request
        only, which is how a caller supplies the side question itself (and any
        in-flight assistant text it is painting but the context does not carry
        yet). No tool is ever executed: the live tool schema rides along only
        to keep the request on the working turn's cache prefix, and a
        ``tool_use`` block in the answer is inert.

        It is NOT free: the request carries the whole conversation. Nothing is
        recorded, so ``on_usage`` reports the provider's own figures to
        whatever the host counts spend with. It fires once per provider call,
        and an aside may make two — when the answer was a bare tool call with
        no text, the implementation retries once without tools — so a host
        must SUM the callbacks, not keep the last; both calls were billed.

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
    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> str | None:
        """Run one user turn to completion (awaitable) or raise.

        ``images`` are attachments pasted into the prompt; they ride the same
        message as the text so the model reads them as one turn.

        ``@path`` references in ``text`` expand BEFORE the turn is built, so
        every surface reaching this method gets them without per-surface work.
        A token that does not resolve to an existing path is not a reference:
        it is prose, and it is left verbatim. There is no provenance keyword to
        pass and the signature does not widen — the resolver's strictness, not
        a flag, is what makes expanding any text safe.

        THE RETURN IS THE OWNER'S OWN RECEIPT, when the transport has one
        (``str | None``): the in-process ``Session`` runs the whole turn and
        answers with nothing, while an ``AttachedSession`` is answered on
        durable admission and passes the owner's sentence back unchanged — so a
        viewer can tell 'the runtime admitted this' from the one other receipt
        a draining owner can give, 'it was queued for the build that replaces
        me'. A front end with nothing to do with the distinction may ignore it,
        which is what every caller did while the line was being dropped here.
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

    def queued_steering(self) -> list[AgentMessage]:
        """A FIFO snapshot of the steering queue, without draining it.

        The read half of the recall seam: hosts deciding what is still
        recallable must be able to see the queue, whose entries keep their
        identity.
        """
        ...

    def steer_message(self, message: Message) -> None:
        """Queue a caller-built steering message, sharing the caller's object.

        The identity-preserving twin of :meth:`steer`, for hosts that keep a
        reference to what they queued so :meth:`recall_steering` can take it
        back.
        """
        ...

    def recall_steering(self, message: AgentMessage) -> bool:
        """Remove ONE specific message from the steering queue, if present.

        Lets a host unsend a queued mid-turn steer — the TUI's Esc lifts the
        newest one back into the composer. Matched by identity (the very
        object ``steer_message`` queued), so equal-but-distinct messages and
        queue entries the host never queued (wake deliveries) are untouched.
        False when the message is not queued — already drained at a boundary,
        or never queued — and changes nothing.
        """
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

    # --- credentials ------------------------------------------------------
    async def credential_op(self, action: str, key: str = "", value: str = "") -> dict[str, Any]:
        """Run one ``/credential`` verb against the session's store.

        A SESSION capability, declared here so it is declared once for every
        session shape: a session that runs its tools in this process executes
        the verb against its own store, and a session that is a window onto a
        runtime routes it there, because the store the tools' environment is
        built from lives beside the turn loop — on the wrong side of that
        split, a stored key would be advertised to the model while no
        executing tool could read it.

        The value is never logged, never journalled, and never returned —
        only the key name and the outcome cross back.
        """
        ...

    # --- code memory (execution variables) --------------------------------
    async def variables_op(
        self, action: str, key: str = "", value: str = "", value_type: str = ""
    ) -> dict[str, Any]:
        """Run one session code-memory verb (list/set/update/delete).

        A SESSION capability for the same reason ``credential_op`` is: the thing
        being read or written is the LIVE eval-kernel namespace, which only
        exists in the process running the session's turn loop. A session that
        runs its tools in this process answers from its own kernel registry; a
        session that is a window onto a runtime routes it there, because a
        namespace held anywhere else would be a copy no cell ever mutates.

        The answer is the frozen envelope (``{ok, state, kernel, variables,
        truncated}`` for a read, ``{ok, state, variable?}`` for a write,
        ``{ok: False, code, message}`` for a refusal) — see
        :mod:`local_operator.session.variable_ops`.
        """
        ...

    # --- events -----------------------------------------------------------
    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register an event handler; returns an unsubscribe callable."""
        ...

    # --- lifecycle --------------------------------------------------------
    async def dispose(self) -> None: ...


@runtime_checkable
class ViewerSessionProtocol(SessionProtocol, Protocol):
    """The surface a VIEWER offers on top of :class:`SessionProtocol`.

    A viewer is a facade attached to a runtime that executes turns somewhere
    else. It cannot run the loop, but it can do things a runtime has no need
    to do: page a display window it did not build, ask the runtime to stop,
    adopt a takeover, report which build the runtime is running.

    **Why this is declared.** These members exist only on ``AttachedSession``,
    and its hosts reached all of them through ``getattr(session, "name", None)``
    duck-probes rather than through a type \u2014 roughly forty undeclared members
    at the time of writing, on the object the entire front end is written
    against. The exact figure is not maintained here: it is derived by
    ``tests/unit/session/test_viewer_protocol.py``, which recomputes the
    viewer-only set from the classes on every run, and quoting a constant in a
    file whose point is that the number is derived is how the constant becomes
    wrong. A rename on the facade or a typo in a probe string is invisible to
    pyright and surfaces as a silently missing capability at runtime, which is
    exactly how ``/info`` reported zero subagents for a session that had
    several (see ``AttachedSession.subagent_comms``).

    **Why it is a separate protocol and not more of SessionProtocol.**
    Runtimes genuinely do not have these members and should not be forced to grow
    no-op stubs for them; a stub that returns a plausible empty value is worse
    than an absent attribute, because the caller cannot tell "nothing to show"
    from "not this kind of session".

    **How the TUI reaches it.** Through ``tui/app.py::_is_viewer``, a
    ``TypeGuard[ViewerSessionProtocol]`` over the declared ``owns_runtime``
    predicate — NOT through ``isinstance`` against this class. The sixteen
    ``isinstance(session, AttachedSession)`` checks that preceded it became one
    predicate, and pyright narrows to this type inside every true branch, so
    the members below are statically checked at each of those call sites.

    **Why ``runtime_checkable`` is still on this class**, even though no
    production path performs an ``isinstance`` against it. The decorator is not
    here to be used at runtime by the app; it is what lets
    ``tests/unit/session/test_viewer_protocol.py`` assert conformance
    STRUCTURALLY — that ``AttachedSession`` satisfies this protocol and that
    ``Session`` does not, which is the property making the split meaningful
    rather than decorative. Those assertions run once per test, never on a
    paint path.

    It is deliberately not used for dispatch, and the reason is measured rather
    than stylistic. This protocol carries 132 public members and a POSITIVE
    ``isinstance`` walks every one of them; measured on an arm64 host, CPython
    3.12.13, min-of-seven over 2,000 iterations:

    The figure is ``len(typing._get_protocol_attrs(ViewerSessionProtocol))`` —
    exactly the set a positive ``isinstance`` walks, which is what makes it the
    right number to quote beside the timing. It counts INHERITED members too,
    so it is larger than the viewer-only population
    ``test_viewer_protocol.py`` pins; the two answer different questions and
    must not be reconciled. It read 84 for some time while the protocol grew
    past it (106 before the warm members were added, 108 with them, 109 once
    ``restored_search_spend`` joined ``restored_usage``, 111 once
    ``can_ever_bind`` and ``session_was_stopped`` joined the viewer contract,
    ``can_ever_bind`` and ``session_was_stopped`` joined the viewer contract,
    112 once the lease-warm retry needed ``recovering``, 113 once
    ``restored_spend`` joined the shared surface a viewer inherits, 114 once
    session code memory joined the session contract with ``variables_op``, 115
    once the retention predicate's clone-free ``has_running_job`` joined the
    per-frame reads, 116 once ``mcp_credentials_op`` joined
    ``ViewerSessionProtocol``, 117 once a move needed the viewer's own ``cwd``,
    118 once the same move needed ``supports_exclusive_move`` so a desktop host
    can fail CLOSED against an owner that would ignore the exclusivity flag, 119
    once it needed the ``set_local_cwd_callback`` seam the move's local
    replacement is published through, 120 once the drain notice's
    ``set_drain_callback`` joined the viewer contract, 121 once the desktop's
    interrupt rung needed ``interrupt`` to stop a turn without ending the
    session, 122 once the same rung needed ``owner_reachable`` so a mid-resync
    viewer could not be read as an absent owner, 124 once a read had to report
    WHY it served cold — ``cold_reason`` and ``attaching``, one rung that adds
    two members because the wire tells a renderer both the fact and the
    in-flight state, 125 once a refused gate reply needed a surface to reach the
    pane that pressed APPROVE, 126 once the operator-prompt notice gave that pane
    the sentence naming what a signature is about to authorise, 127 once a
    paint-first open needed ``attach_behind`` to narrate the attach behind its
    paint, 128 once an answer that was accepted at the pane and never reached its
    owner needed a surface to say so (``set_gate_undelivered_handler``), 129
    once the dock band needed ``frontend_revision`` to skip a roster
    re-derivation nothing moved under, 130 once the desktop withdrawal needed
    ``withdraw_desktop_watch``, 132 once an honest status needed the pair the
    not-stale contract turns on — ``verified_at`` (when the owner last ANSWERED)
    and ``verify_live`` (the ``ping`` round trip that refreshes it), which is one
    rung adding two members because a stamp with no way to refresh it cannot
    distinguish "the owner is there" from "the owner was there"), so
    recompute it rather
    than adjusting it by the size of your own change.

    ====================================================  ==================
    ``isinstance(viewer, AttachedSession)`` (what it was)    0.014-0.015 us
    ``isinstance(viewer, ViewerSessionProtocol)``            55-58 us
    ``not session.owns_runtime`` (what it is now)          0.021-0.024 us
    ====================================================  ==================

    THE ABSOLUTE IS THE PORTABLE FACT; the ratio is an order of magnitude and
    should be quoted as one (~10^3x). The positive reproduced across three
    independent hosts at 55-58 us, but the same three computed ratios of
    ~2,400x, ~1,500-1,690x and ~930-1,054x from it — because the sub-100 ns
    denominator is dominated by the timing loop rather than by the work. One
    host measured an empty-lambda floor of 0.018 us, 34% of its own predicate
    reading, and floor-subtracting moved its ratio to ~1,641x. So a narrow band
    here is a property of whoever last ran it: state the microseconds, which
    travel, and let the ratio carry only the magnitude, which is all the
    decision needs. (The negative case is cheap, ~1-3 us: it fails on the first
    missing member. Only the positive path pays, and the sites below are
    overwhelmingly positive.)

    **HOW to re-measure, because the obvious method silently measures the wrong
    path.** Only a FULLY CONSTRUCTED ``AttachedSession`` is a positive here — the
    cost is 84 property getters actually executing, so anything cheaper to build
    is cheaper to check for the wrong reason:

    ==========================================  =========  ==========
    construction                                positive?  measured
    ==========================================  =========  ==========
    ``AttachedSession(config_dir=..., ...)``       **yes**    ~55 us
    ``AttachedSession.__new__(AttachedSession)``     no         ~15 us
    ``MagicMock(spec=AttachedSession)``            no         ~3 us
    a synthetic class from ``__protocol_attrs__``  no       ~1-2 us
    ==========================================  =========  ==========

    Each of the bottom three looks like a viewer and reports a figure one to two
    orders of magnitude low, because it is timing the negative path that exits
    on the first missing member. So **assert the positive before timing it** —
    ``assert isinstance(v, ViewerSessionProtocol)`` — or the number produced is
    an answer to a different question. This cost a round-1 reviewer three
    attempts and produced one re-measurement (~7.5 us) that could not be
    reproduced by two others; the earlier 60-136 us figure quoted here was
    likewise never reproduced and has been replaced by the measurement above.

    Three of the converted sites are hot — the sidebar release sweep runs per
    source per pass, ``_relaunch_refusal`` loops every interaction — so
    dispatching on the type would put three orders of magnitude on a paint path
    to buy a guarantee the ``TypeGuard`` already provides statically.

    Headless hosts need this surface too, which is why it lives here rather
    than in the TUI: ``server/utils/desktop_sessions.py`` already imports,
    types against, and constructs ``AttachedSession`` to serve history pages.

    The member list is not curated by hand \u2014 it is derived by AST from every
    session-valued attribute access in ``tui/app.py`` and asserted complete by
    ``tests/unit/session/test_viewer_protocol.py``, which fails when a new
    undeclared duck-typed member appears.
    """

    # --- what kind of viewer this is --------------------------------------
    @property
    def is_cold(self) -> bool:
        """Whether NO synchronised runtime is reachable through this viewer.

        True in two distinct situations, and callers depend on it covering
        BOTH:

        * never bound \u2014 a session id and a transcript but nothing executing:
          `lop` launched, the user opened the picker, nothing typed;
        * bound and now unreachable \u2014 the socket dropped, the runtime
          died, or the facade is redialing a replacement.

        The implementation is "no client, or not connected, or not ready for
        events", so the second case is deliberate rather than incidental.
        ``app.py:23470`` states the dependency outright ("``is_cold`` is a
        superset of the drop: it is also true while the facade is redialing a
        runtime that died"), and narrowing it to "never bound" would silently
        re-open the #625 shape at that site.

        So it does NOT distinguish never-bound from lost. A caller needing
        that reads ``degraded_reason`` or ``runtime_pid``; a caller asking
        "is a turn reachable right now" wants exactly this.
        """
        ...

    @property
    def verified_at(self) -> float | None:
        """When a round trip to this viewer's owner was last ANSWERED, or ``None``.

        THE EVIDENCE TERM BESIDE :attr:`is_cold`, and it exists because that
        predicate cannot carry the claim on its own: it is three local reads with
        no round trip in it, so a viewer whose owner has been SIGSTOPped still
        reads live for as long as its socket stays open — the kernel keeps the
        connection and nothing observable locally changes. A host that publishes
        "this session is live" must read this, not just ``is_cold``.

        ``None`` means NEVER VERIFIED and is deliberately distinct from stale: a
        cold viewer carries no stamp at all. Nothing on the writer side expires
        it — a reader applies its own age budget
        (``runtime.types.LIVE_FRESHNESS_BUDGET_S``) — so an owner that dies
        mid-update leaves a stamp that stops advancing rather than a fresh false
        "current".

        NOT a replacement for :attr:`owner_reachable` ("is there an owner to
        ask") or for the ``_ready_for_events`` sync term, and no resync clears
        it: clearing one on a display refresh would turn a live, mid-refresh
        viewer cold, which is the same class of lie in the other direction.
        """
        ...

    async def verify_live(self, timeout: float) -> bool:
        """Ask the owner to answer, and stamp the verification if it does.

        The instrument is the attach protocol's existing ``ping`` op, tagged
        *"list — liveness only"* and exempt from the owner's op chain so a health
        probe is answered while mutations queue. Returns ``True`` when the owner
        answered inside ``timeout``.

        An unanswered probe returns ``False`` and TOUCHES NOTHING: the previous
        stamp stays where it was, so a failure narrows the staleness window
        instead of clearing it — a clear would read as a fresh classification
        and turn a merely-slow owner cold.
        """
        ...

    @property
    def can_ever_bind(self) -> bool:
        """Whether a bind attempt on this viewer could EVER succeed.

        Its own question, and not a refinement of :attr:`is_cold`. A facade is
        cold for many reasons that all clear on their own — a socket blip, an
        owner loss the recovery loop is already chasing, a never-bound row that
        has not been asked to dial yet, a live owner whose display history is
        mid-refresh — and every one of those binds or syncs. This asks the one
        thing a caller cannot wait out: whether the viewer is closed to dialling
        BY CONSTRUCTION, with nothing reachable on the other end.

        It is therefore NOT the negation of the guard at the top of
        ``AttachedSession._ensure_bound``, which returns in states this answers
        True for (a mid-refresh client, a facade with a recovery loop running).
        False is that guard with no live owner behind it:

        * the LEGACY attach contract (``_can_go_cold`` false, what ``connect``
          builds unless a caller asks for the viewer contract) once its owner is
          gone. That facade never dials, and what normally releases it is its
          own recovery loop (``_give_up_recovery`` sets the flag before going
          cold). A DELIBERATE stop is both the arm where no loop ever runs —
          ``_on_disconnected`` returns before starting one, and
          ``_recover_runtime`` returns instead of giving up when its own wake
          marker says the session was stopped — and the arm where nothing sets
          the flag, so the state is permanent;
        * a DISPOSED facade, which refuses every path.

        ``_recovering`` must answer True, and the distinction is load-bearing:
        the flag means "a recovery loop is running", and every exit of that loop
        either attaches this facade or releases it rebindable, so the state is
        transient by construction. A caller that treated it as final would tell
        the user that a session on its way back is gone.

        **Why this is DECLARED rather than probed.** The caller that needs it
        uses the answer to choose a sentence the user reads, which is exactly
        where a ``getattr(session, "_can_go_cold", None)`` duck-probe belongs:
        that is how the original sidebar defect shipped — a private read nothing
        type-checks, and a rename on the facade could not break it loudly. The
        name answers the question the caller has, not the flag it reads, so a
        future second reason to be un-bindable does not put a second name in
        every host.
        """
        ...

    @property
    def runtime_pid(self) -> int | None:
        """Pid of the runtime this viewer is attached to.

        ``None`` whenever ``is_cold`` is true \u2014 including after a runtime this
        viewer WAS attached to died, since the disconnect hook clears it. It
        is "no runtime reachable now", not "never had one".
        """
        ...

    @property
    def can_detach_runtime(self) -> bool:
        """Whether leaving this session would leave a runtime running."""
        ...

    # --- the runtime on the other end -------------------------------------
    #: The BUILD the runtime is running, read off its discovery record at
    #: dial. ``""`` on a facade that has never bound AND on one bound to a
    #: runtime older than the field; hosts distinguish those by ``is_cold``,
    #: not by this value.
    runtime_version: str
    #: The source ref of that build, same lifecycle as ``runtime_version``.
    runtime_source_ref: str
    #: Why this viewer opened WITHOUT live state, when that was not the
    #: ordinary "no runtime was running" case. ``""`` when it was ordinary.
    degraded_reason: str
    #: The launcher opened this viewer cold in FRONT of a live owner it binds to
    #: behind the paint (paint-first `lop --resume` / `/resume`), so the TUI
    #: narrates and bounds that attach. Viewer-only: an owner has no owner to
    #: attach to.
    attach_behind: bool
    #: Whether the saved preview this viewer opened against was partial.
    saved_preview_partial: bool

    def runtime_idle(self) -> bool:
        """Whether the runtime reports no turn in flight."""
        ...

    def runtime_model_catalogue(self) -> list[dict[str, Any]]:
        """The model catalogue as the RUNTIME resolves it.

        A viewer must not answer from its own machine's catalogue: the runtime
        may hold different credentials and therefore offer different models.
        """
        ...

    async def attach_existing(self, *, budget: float | None = None) -> bool:
        """Bind to a runtime if one is already live, without starting one.

        The desktop host calls this on a cold viewer so that serving a history
        read never promotes a reader into an executor: losing the runtime must
        not move execution into the HTTP worker.

        ``budget`` selects READ MODE: a read gets one attempt bounded by that many
        seconds and is then answered from disk rather than refused, with the
        authenticated dial retained in case the owner's canonical state lands
        afterwards. ``None`` keeps the CONTROL envelope every other caller has,
        where the same failure is a raise.
        """
        ...

    async def session_was_stopped(self) -> bool:
        """Whether this session ended DELIBERATELY rather than dying.

        A host that watched the disconnect can classify it from the wire, but a
        host that did not — the sidebar re-dials a row clicked later — has only
        this: the durable ``stopped_at`` marker the stop path stamps, plus this
        viewer's own ``request_stop``. The question it answers is the one that
        decides whether re-dialling is worth anything, so the answer must live
        here rather than be re-probed by each front end.

        FALSE IS "NOT PROVEN STOPPED", not "proven alive", and the difference
        matters to a caller composing a verdict: the marker is written by
        ``control._mark_wakes_dormant``, which writes NOTHING for a session with
        no wake schedules (an absent index file is the store's own "no wakes"),
        and it is cleared when the session is next opened. A wake-less stop
        therefore leaves no trace here at all.
        """
        ...

    async def update_desktop_watch(self, *, visible: bool, can_notify: bool) -> None:
        """Renew this viewer's desktop attach lease.

        Desktop-surface viewers only; the host recomputes visibility and
        notifiability from its live subscribers and pushes the result so a
        bare proxy socket is not mistaken for a watching human.
        """
        ...

    async def withdraw_desktop_watch(self) -> None:
        """Withdraw this viewer's desktop attach lease: the pane has left.

        Desktop-surface viewers only. The explicit end-of-attachment signal
        (the transport-bound hole): the runtime CLEARS its session-scoped
        attach memory rather than renewing it, and the facade keeps the
        withdrawal as the state a re-dial replays — so a successor runtime
        starts detached instead of resurrecting a lease nobody holds. Its own
        member rather than a shape of ``update_desktop_watch`` because no
        ``(visible, can_notify)`` pair can carry the meaning: a transient
        renderer stream end sends ``(False, False)`` too, and must keep the
        memory alive.
        """
        ...

    @property
    def frontend_state(self) -> "FrontendSessionState":
        """The canonical snapshot this viewer renders from.

        Declared HERE rather than on :class:`SessionProtocol`, even though both
        session classes implement it, and the reason is measured rather than
        stylistic: ``SessionProtocol`` is what every test double and reduced
        host is checked against, so adding a member there made **1,871** pyright
        errors across the suite for doubles that legitimately do not carry
        canonical state. The viewer protocol is narrower — only the real facade
        and the desktop bridge satisfy it — so the declaration lands exactly
        where the hard accesses are.

        Undeclared debt (``_UNDECLARED_ON_BOTH_CLASSES``) until Stage 3's
        type-narrowing made pyright ask for it at three TUI sites. That is the
        guard working: the member was always reachable and always unchecked.

        **Raises until the first sync.** ``AttachedSession.frontend_state`` raises
        ``RuntimeError`` while ``_frontend_store`` is None, so a caller on a
        paint path that cannot tolerate a raise probes defensively instead —
        several in ``app.py`` deliberately do, and ``_session_subject`` records
        what a raise there costs.
        """
        ...

    @property
    def supports_exclusive_move(self) -> bool:
        """Whether the bound owner can retire under the move exclusivity fence.

        Asked by the desktop move path BEFORE it mutates anything, because an
        owner without ``exclusive-move-v1`` ignores the ``exclusive`` field and
        retires unguarded — leaving a sibling facade to engage a successor from
        its own stale cwd. False means refuse with update guidance; it never
        means "fall back to a plain retire".
        """
        ...

    @property
    def supports_completion_ack(self) -> bool:
        """Whether the attached runtime can acknowledge completion attention.

        Read by the DESKTOP host rather than the TUI: it decides whether the
        phone portal is told completion-attention is supported. A runtime too
        old to carry the ack answers ``False``, which is why the host pairs
        this with ``is_cold`` instead of reading absence as a capability.
        """
        ...

    # --- display history paging -------------------------------------------
    # A viewer renders a window over a transcript it does not own, so paging is
    # a request to the runtime rather than a slice of a local list. The revision
    # counter is what lets the TUI tell "same window, redrawn" from "the window
    # moved" without diffing rows.

    @property
    def display_history_revision(self) -> int:
        """Bumped whenever the display window's CONTENT changes."""
        ...

    @property
    def display_history_current(self) -> bool:
        """Whether the loaded window includes the newest message."""
        ...

    @property
    def history_message_count(self) -> int:
        """How many messages the runtime's transcript holds in total."""
        ...

    @property
    def history_before_token(self) -> str | None:
        """Opaque cursor for the next older page; ``None`` at the beginning."""
        ...

    @property
    def history_is_audit(self) -> bool:
        """Whether the rows still above the reader are PRE-COMPACTION history.

        The head notice branches its copy on this: rows behind the compaction
        cut are real history the model can no longer see, which is a different
        kind of row from "older messages" rather than merely an earlier one.
        ``False`` on a runtime too old to page them, so that viewer keeps the
        copy it always had.
        """
        ...

    @property
    def history_opener_text(self) -> str:
        """The first user message's text, for the session's title row."""
        ...

    @property
    def history_theme_turn_count(self) -> int:
        """Turns available to the naming pass."""
        ...

    def display_history_window(self) -> list[Any]:
        """The currently loaded window of display rows."""
        ...

    def history_last_message(self) -> Any:
        """The newest message in the loaded window, or ``None``."""
        ...

    def pending_display_tool_ids(self) -> set[str]:
        """Tool ids whose result rows have not arrived yet."""
        ...

    def executing_display_tool_ids(self) -> set[str]:
        """Tool ids of a STILL-RUNNING turn, so a replayed row is not settled.

        The gate's counterpart: a long tool parks the turn inside execution
        with no gate open, so ``pending_display_tool_ids`` is empty while the
        call is alive and a replay would paint ``⊘ interrupted`` over it.
        Viewer-side for the same reason as its sibling above — it answers a
        display question off ``is_streaming`` plus the latest call group, and
        has no meaning on a runtime nobody is viewing.
        """
        ...

    def live_tool_start_epochs(self) -> dict[str, float | None]:
        """Start instant per call executing RIGHT NOW, keyed by call id.

        The timestamped sibling of :meth:`executing_display_tool_ids`, and the
        one thing that makes a live row's elapsed clock survive a change of
        viewer. Both surfaces answer it off the SAME folded fact — the
        ``tool_execution_start`` events the producer emitted (see
        ``ToolExecutionStartEvent.started_at_epoch``) — so a sidebar switch
        seeds the replayed row and the band's phase from one anchor rather
        than from whenever each of them was painted.

        Declared here beside the id accessors it accompanies, and implemented
        by BOTH session shapes: a local owner answers from its own fold (it is
        the producer, so the instants are its own), and an attached viewer
        answers from the same fold applied to the events it received.

        Read membership and value as two different answers, because a replay
        has to ask both:

        * MEMBERSHIP — has this call started? A call that has no entry at all
          is one the tail scan merely cannot pair with a result yet: queued
          behind a sibling's execution group, or never run. Its row must not be
          painted as executing. Membership is what distinguishes that from the
          far more common case below, and the distinction is why the map is not
          simply a list of epochs.
        * VALUE — when it started, or ``None`` when the start carried no epoch
          (a legacy producer, an older runtime). An epoch-less start is present
          with ``None`` rather than absent: the event DOES say the call began,
          and the value is withheld instead of guessed, so consumers keep the
          clock blank rather than printing an age nobody measured.

        An empty map is therefore a valid, supported answer and not an error.
        """
        ...

    def activity_phase_clock(self) -> tuple[str, float | None]:
        """The folded working-line phase, and the instant that phase began.

        The companion :meth:`live_tool_start_epochs` cannot supply, and the
        reason the operator's report names TWO clocks rather than one: the
        thinking indicator has no tool call behind it, so there is no id to key
        an epoch by and nothing for a per-call map to answer with. The phase is
        what the band's number is anchored to, so the phase's own start is what
        has to survive a switch.

        Read together on purpose. The consumer's rule is "use this instant only
        when this phase is the phase I just derived", and two independent reads
        could pair one phase with the previous phase's zero — a wrong age that
        would look perfectly plausible. A facade with no fold answers
        ``("", None)``, which matches nothing and therefore withholds the
        clock rather than inventing one.
        """
        ...

    async def ensure_display_anchor(self, anchor: str) -> bool:
        """Load whichever page contains ``anchor``; False when it is gone."""
        ...

    async def ensure_display_current(self) -> None:
        """Move the window to the newest page."""
        ...

    async def load_older_display_page(self) -> list[Any]:
        """Extend the window one page toward the beginning."""
        ...

    async def materialize_history(self) -> list[Any]:
        """Pull the WHOLE transcript, not just the loaded window.

        Expensive by construction \u2014 for the paths that genuinely need every
        message (export, search, compaction preview), not for rendering.
        """
        ...

    # --- driving the runtime -----------------------------------------------
    async def prompt_and_wait(
        self,
        text: str,
        images: Sequence[ImageContent] | None = None,
        *,
        message_id: str | None = None,
    ) -> None:
        """Submit a turn and wait for its terminal outcome.

        The counterpart to :meth:`SessionProtocol.prompt`, which on a viewer
        returns on the runtime's admission ACK rather than on the outcome (see
        :attr:`SessionProtocol.outcome_is_synchronous`). A host that needs the
        result \u2014 exec mode, a scripted turn \u2014 must await THIS.
        """
        ...

    async def bind_runtime(self) -> None:
        """Bind this viewer before an explicitly requested runtime operation.

        Declared because the desktop ROUTES call it as a HARD access on every
        path that mutates runtime state (``desktop_lifecycle.py`` /mcp,
        /credential, /fork, aside completion and adoption;
        ``desktop_sessions.py`` routed slash), reached through the
        ``bridge.remote`` binding. A rename on the facade is therefore an
        ``AttributeError`` inside a live HTTP route rather than the silent
        ``None`` a duck-probe would return — louder, but a 500 on the phone
        portal all the same (QA round 2, Q4).

        Unlike the duck-probed members, these three were NOT invisible to
        pyright before this declaration: the routes reach them through
        ``DesktopSessionBridge.remote``, annotated as a concrete
        ``AttachedSession | None``, so a rename was already a type error there —
        reproduced against the pre-PR base (QA round 3, Q8, correcting an
        earlier claim here that it was not). Declaring them buys the protocol's
        coverage of the surface, not a check that was missing.

        Distinct from :meth:`attach_existing`: this one is allowed to START a
        runtime, because the caller is a user action that needs one. A READ
        must use ``attach_existing`` so serving history never promotes the HTTP
        worker into an executor.
        """
        ...

    async def admit_prompt(
        self, text: str, *, command_id: str, images: list[dict[str, str]], steer: bool = False
    ) -> tuple[str, bool]:
        """Submit a prompt and return the runtime's ADMISSION receipt.

        ``(detail, duplicate)``: the runtime's receipt text, and whether the
        stable ``command_id`` matched a reservation it already holds. It does
        not wait for the turn's outcome — that is the point. The desktop routes
        serve an HTTP request, and an HTTP disconnect must not cancel work the
        runtime already accepted, which is what awaiting completion would allow.

        Same declaration reasoning as :meth:`bind_runtime`: a HARD access from
        ``desktop_lifecycle.py`` and ``desktop_sessions.py``.
        """
        ...

    async def answer_gate(
        self,
        request_id: str,
        *,
        value: str | None = None,
        approved: bool | None = None,
        question_index: int | None = None,
    ) -> str:
        """Answer the runtime's open approval gate; returns its receipt.

        ``request_id`` is checked against the gate the facade currently holds
        before anything crosses the wire, so a stale desktop popup cannot
        answer a NEWER gate that a reconnect or a multi-question ask advanced
        in another window. The runtime validates again on its side; this check is
        what keeps the wrong answer from being sent at all.

        Same declaration reasoning as :meth:`bind_runtime`: a HARD access from
        ``desktop_sessions.py``.
        """
        ...

    async def request_stop(self) -> str:
        """Ask the runtime to end the session; returns its receipt."""
        ...

    async def request_refresh(self) -> str:
        """Ask the runtime to re-send canonical state."""
        ...

    async def retire_if_unused(self) -> str:
        """Ask the runtime to retire itself if nothing else is attached."""
        ...

    async def set_working_directory(self, cwd: str) -> str:
        """Change the RUNTIME's working directory; returns its receipt."""
        ...

    async def route_shared_slash(
        self,
        command: str,
        args: str,
        images: Sequence[ImageContent] | None = None,
    ) -> Any:
        """Run a slash command on the runtime and return its result."""
        ...

    async def mcp_credentials_op(self, body: dict[str, Any]) -> dict[str, Any]:
        """Write declared MCP credential values to the RUNTIME's encrypted store.

        Declared for the same reason as :meth:`warm_runtime` below: the desktop
        bridge reaches it through ``bridge.remote`` on
        ``POST /v1/desktop/sessions/{id}/mcp/credentials``, so a rename here has
        to be a type error rather than a silently missing capability — this is the
        route that stores the key a UI-added MCP server authenticates with.

        A VIEWER forwards it to the owner rather than writing locally: the store
        it resolves against is the one the runtime's connects read, and a value
        written anywhere else would be a copy no server ever sees. ``body`` is the
        validated ``MCPCredentials`` payload (``name``/``values``/
        ``confirmed_replace``), and the answer is the frozen envelope
        (``{code, saved_ids, failed_ids, name}``) — never the value, which is why
        the shape carries ids only. Raises when there is no owner to write
        through, unlike the read-side probes that report a disconnected state.
        """
        ...

    def move_will_wait(self) -> bool:
        """Whether a move would block on an in-flight turn."""
        ...

    @property
    def cwd(self) -> str:
        """Where this session works, i.e. what :meth:`set_working_directory` moves.

        Declared for the same reason as :attr:`engage_in_flight` below: the move
        route resolves a relative target against this value and compares against
        it to decide a no-op, so a rename on the facade must be a type error
        rather than a silently stale base for every path a user types.
        """
        ...

    async def warm_runtime(self) -> None:
        """Engage a runtime speculatively, for a caller nobody is waiting on.

        Declared for the same reason as :meth:`bind_runtime` beside it: the
        desktop bridge reaches it through ``bridge.remote`` on the /warm route,
        so a rename on the facade must be a type error rather than a silently
        missing capability on the hottest new path in the desktop app.

        Never raises, and is NOT interchangeable with :meth:`bind_runtime`: it
        takes the background bind envelope and is silent on failure, because a
        warm-up the user did not ask for must never become an error they have
        to read.
        """
        ...

    @property
    def engage_in_flight(self) -> bool:
        """Whether an engage is running that another caller would have to join.

        A HINT, not a guarantee — it samples a lock at one instant. Declared
        because the desktop bridge reads it to decide whether a speculative
        warm needs starting at all; correctness under a missed sample belongs
        to the bind lock, not to this predicate.
        """
        ...

    @property
    def recovering(self) -> bool:
        """Whether owner recovery owns this viewer's dial right now.

        Declared for the same reason as :attr:`engage_in_flight` beside it, and
        it answers the question that predicate cannot: an engage attempted while
        recovery owns the dial does NO WORK at all — ``_ensure_bound`` returns at
        its own guard, with no task to report and no process to account for — so
        the desktop bridge's lease-driven warm reads this to tell a REFUSED
        attempt (retried on a short poll) from a FAILED one (which it paces,
        because that attempt really did spawn). Unlike the lock sample above,
        this is a state read: there is no window between asking and acting.
        """
        ...

    # --- job trajectories --------------------------------------------------
    async def load_job_trajectory(self, job_id: str) -> bool:
        """Stream a subagent's transcript into this viewer; False if absent."""
        ...

    async def unload_job_trajectory(self, job_id: str) -> None:
        """Drop a loaded subagent transcript."""
        ...

    # --- approval gates across a detach ------------------------------------
    # A viewer that leaves must not strand a gate the user never answered, and
    # one that returns must not replay a gate the runtime already resolved.

    def suspend_viewer_gates(
        self, *, auto_approve: bool = False, keep_answer: bool = False
    ) -> "asyncio.Task[Any] | None":
        """Park this viewer's open gates; returns the task settling them."""
        ...

    def resume_viewer_gates(self) -> None:
        """Re-arm gates after a suspension that did not detach."""
        ...

    async def detach_viewer_gates(self, *, preserve_answers: bool = False) -> None:
        """Settle open gates because this viewer is leaving for good."""
        ...

    def preserve_viewer_gate_reply(self) -> None:
        """Latch an answer the user committed before its bridge is scheduled.

        Called synchronously by the UI as it settles a gate's future: waiting
        for the wire send loses answers when a reload lands on the same tick.
        """
        ...

    @property
    def has_pending_gate_reply(self) -> bool:
        """Whether a latched answer is still in flight to the runtime.

        The sidebar reads this to avoid tearing down a source whose committed
        reply has not reached the runtime yet. Probed as a 3-arg ``getattr``
        defaulting to ``False`` before this declaration, so a rename on the
        facade silently stopped protecting the reply rather than failing.
        """
        ...

    async def interrupt(self) -> str:
        """Stop the owner's CURRENT TURN and return its receipt, verbatim.

        VIEWER-ONLY BY CONSTRUCTION, which is why it is not on
        :class:`SessionProtocol` beside :meth:`abort`: an owner ``Session`` stops
        its own turn with a local call and has nobody to ask, while a viewer
        dials the owner's ``abort`` control frame and is handed a sentence back.
        The desktop route reads this off a duck-typed bound facade to answer the
        Stop button, so the name must be declared here — an undeclared read
        degrades to a silent ``None`` and a rename would answer the press with
        nothing rather than failing.

        Not the kill switch (:meth:`request_stop` ends the session and its
        process); this ends one turn and leaves both running.
        """
        ...

    @property
    def owner_reachable(self) -> bool:
        """Whether a live owner exists to ask — reachability, not sync state.

        Distinct from :attr:`is_cold`, whose third disjunct (``not
        _ready_for_events``) is a RESYNC state that is true of a connected,
        serving session mid-refresh; a caller deciding whether to dial must not
        read a mid-resync viewer as an absent owner. Declared here because the
        desktop interrupt route reads it off a duck-typed bound facade to choose
        between an ``idle`` answer and dialling the owner.
        """
        ...

    @property
    def attaching(self) -> bool:
        """Whether an authenticated dial is retained, waiting for its sync.

        The desktop bridge puts this on the wire (``snapshot.payload.attaching``)
        so a renderer can tell "the runtime is gone" from "the runtime has
        accepted us and its state has not arrived yet" — the second is an
        ordinary busy loop, and reading it as the first is what made a read
        refuse a session that was running. Declared because the read reports it
        off a duck-typed facade; an owner ``Session`` has no dial at all, so the
        state does not exist for it.
        """
        ...

    @property
    def cold_reason(self) -> str | None:
        """WHY a cold facade is cold, as one of the three wire tokens, or None.

        ``"no-runtime"`` (no pid holds the session's transcript lease),
        ``"owner-silent"`` (one does, and did not deliver canonical state) or
        ``"owner-leaving"`` (the record is finishing work in flight first). The
        read routes report it so the renderer can act on the difference instead
        of treating every cold frame as a lost conversation; ``None`` means the
        frame is live. Declared here for the same reason as :attr:`attaching` —
        the read reports it off a bound facade, and the question is meaningless
        for an owner session.
        """
        ...

    # --- runtime-lifecycle callbacks ---------------------------------------
    # Installed once at adoption. They are how a viewer learns about outcomes
    # it cannot poll for: the runtime died and this process won the lease, the
    # runtime ended the session, the runtime retired itself for a newer build.

    def set_takeover_callback(self, callback: Callable[[Any], Any]) -> None:
        """Called with a runtime-owning session when this process wins the lease."""
        ...

    def set_stopped_callback(self, callback: Callable[[], Any]) -> None:
        """Called when the runtime ENDED the session deliberately."""
        ...

    def set_refresh_callback(self, callback: Callable[[], Any] | None) -> None:
        """Called when the runtime retired itself for a newer build."""
        ...

    def set_drain_callback(self, callback: Callable[..., Any] | None) -> None:
        """Called the moment the runtime announces a departure that REFUSES work.

        The sibling of :meth:`set_refresh_callback` one event earlier: that one
        fires when the socket closes and re-engages, this one fires on the
        ``retiring`` frame while the runtime is still working, and only when
        the frame says the departure is draining. Viewer-only by construction —
        an owner ``Session`` has no wire to hear the frame on.

        ``callback`` takes the frame's ``leaving`` phrase: the trigger's own
        words (a signal and a replaced build are two different sentences to a
        reader). It is EMPTY only when the frame named no trigger at all — not
        when the runtime predates the key, which is the older rule and the wrong
        one: a runtime that predates ``leaving`` still hands the signal phrase
        over for a signal drain, because its frame's ``reason``/``to`` decide
        (``types.drain_phrase_for_frame``; agent review round 5, MINOR-2).
        """
        ...

    def set_gate_refusal_handler(self, handler: Callable[[BaseException], None] | None) -> None:
        """Install the surface that reports a gate reply the OWNER refused.

        Viewer-only by construction: an owner ``Session`` answers its own gates,
        so there is nobody above it to refuse an answer. On a facade the refusal
        arrives AFTER the approval handler has returned — the pane pressed
        APPROVE and the runtime declined it because this connection is not the
        window that started the session (issue #1310) — and before this hook the
        refusal had nowhere to go and simply vanished (design review round 2,
        D9; the host that reads it is ``tui/app.py``).

        Declared here rather than probed with ``getattr`` like its two siblings
        on this class because the app calls it on every attached session: an
        undeclared duck-typed member is what
        ``tests/unit/session/test_viewer_protocol.py`` exists to catch, and it
        did catch this one (agent review round 3, Q5 — the head was red).
        """
        ...

    def set_gate_undelivered_handler(self, handler: GateUndeliveredHandler | None) -> None:
        """Install the surface that reports a gate reply the owner never RECEIVED.

        Viewer-only for the same reason as the refusal hook above, and the reason
        is worth stating in its own words because the two are easy to conflate:
        an owner ``Session`` answers its own gates in-process, so an answer given
        here has no wire to cross and cannot fail to arrive. On a facade the
        answer is POSTED to an owner after the host's handler has returned, and
        that post can find the owner gone — a stop landing under a live card is
        the reachable shape (UX review round 1, U1). ``handler`` receives the
        gate's kind, ``"ask"`` or ``"approval"``, and the gate's identity,
        ``(kind, request_id, question_index)`` or ``None`` — the host needs the
        second to retract the receipt THAT card wrote rather than whichever
        receipt happens to be retained (agent review round 3, A12).

        Declared here rather than probed with ``getattr`` like the two optional
        siblings on this class because the TUI calls it on every attached
        session, which is the same test the refusal hook failed before it was
        declared (``tests/unit/session/test_viewer_protocol.py``).
        """
        ...

    def set_local_cwd_callback(self, callback: Callable[[str], Any] | None) -> None:
        """Called when a move installs an accepted directory locally.

        Viewer-only by construction: the callback exists so a DESKTOP bridge can
        publish its own ``frontend.replace`` frame instead of the facade
        emitting a same-sequence delta the renderer discards. An owner
        ``Session`` has no host above it to repaint, so it must not grow one.
        """
        ...

    def set_cancel_resolution(self, resolver: Callable[[int], None] | None) -> None:
        """Arm the seam that rewrites an optimistic cancel count.

        A follower's Esc reports the count it can see synchronously; the
        authoritative number resolves on the runtime and arrives later.
        """
        ...

    def set_recall_resolution(self, resolver: Callable[[str], None] | None) -> None:
        """The recall twin of :meth:`set_cancel_resolution`."""
        ...

    def set_operator_prompt_notice(self, handler: Callable[[str], None] | None) -> None:
        """Arm the sink that paints "what this signature is about to authorise".

        Viewer-only for the same reason the three above are, and one more: the
        sentence describes a PRESENCE PROMPT this machine's key is about to raise,
        and the only surface with a human standing at it is an attached pane. An
        owner ``Session`` runs the loop where the prompt is raised and has no host
        above it to paint on.
        """
        ...

    def set_steer_failure(self, resolver: Callable[[str], None] | None) -> None:
        """Called with the id of a queued steer whose bind was refused.

        The third asynchronous refusal, and the only one with no sender to
        report it: ``steer_message`` spawns a task nobody awaits, so a message
        the app has already echoed as sent can fail without the user ever
        learning. See ``AttachedSession._send_steer_when_ready``.
        """
        ...

    # --- engine state a viewer host renders --------------------------------
    #
    # The MCP status segment and its menus, the subagent and job views, the
    # sidebar's gate identity, the wake panel, the ``/agent`` and ``/team``
    # listings, and the saved-usage band. These live on THIS protocol rather
    # than on ``SessionProtocol`` because that is where the duck-typed readers
    # land. BOTH
    # classes implement them — the owner exposes the live manager, scheduler or
    # registry, and a viewer a read-only SNAPSHOT the runtime published — and
    # every host that reaches them through a DUCK-TYPED binding holds an
    # attached facade: the TUI's ``self._session`` / ``source.session``, the
    # desktop routes' ``bridge.remote``, and ``info/collect.py``'s ``session``
    # parameter.
    #
    # The OWNER-side readers of these same members are NOT an exception to that.
    # ``harness/subagent.py`` types its ``parent_session`` as ``Session`` and
    # reads ``mcp_manager`` / ``mcp_startup`` / ``jobs`` off it, so pyright is
    # already checking those against the real class.
    #
    # ``session/runtime/serving.py`` reads ``self._session.jobs`` on the owner
    # too, and those reads are UNCHECKED: ``ServingSessionHandle.__init__``
    # takes ``session: Any``, so ``self._session`` is ``Any`` there and pyright
    # verifies nothing — not against the real class, not against a protocol.
    # Restating the difference matters because it is the reason a declaration on
    # ``SessionProtocol`` would buy those readers nothing either: what would
    # check them is typing the handle's constructor, not widening the protocol.
    # Either way the placement is a claim about the DUCK-TYPED population and
    # not the claim "only a viewer has these".
    #
    # Declaring them is deliberately NOT the same as making them safe for any
    # host to read. A snapshot member answers from the last sync, so a caller
    # that needs the live object — or the owner-only extra argument, as
    # ``subscribe_frontend``'s ``display_window`` is — must hold the concrete
    # engine type. What the declaration buys is that a rename, a typo or a
    # deletion is now a pyright error and a guard failure instead of the silent
    # ``None`` that shipped a fabricated zero-subagent ``/info``.
    #
    # The wider-protocol claim is not hypothetical: declaring these on
    # ``SessionProtocol`` would add only conformance obligations — the duck-typed
    # hosts read the same members either way — and it put the whole
    # ``tests/unit/tui`` double population off conformance (measured: 213
    # pyright errors across 19 files, from ~10 reduced local ``FakeSession``
    # classes alone; 1996 when every member of the old exclusion list went
    # there). A double that has to grow an MCP manager to keep compiling is a
    # double describing an object the hosts never read. The viewer protocol is
    # the narrower claim, it is true of every duck-typed reader above, and it
    # costs the doubles nothing.

    # Job, wake and subagent plumbing. Read by the subagent view, the wake
    # panel and ``/fork``'s "cannot leave work running" check.

    @property
    def jobs(self) -> Any:
        """Background-job ledger: the live manager, or the runtime's snapshot."""
        ...

    @property
    def wake_scheduler(self) -> Any:
        """Armed wakes: the live scheduler, or the runtime's snapshot."""
        ...

    @property
    def subagent_comms(self) -> Any:
        """Channel to launched subagents: live on an owner, a view on a viewer."""
        ...

    # MCP status and its menus. Both are optional by design: a session may
    # carry no manager at all, and the startup outcome is None until one runs.

    @property
    def mcp_manager(self) -> Any | None:
        """The MCP manager: live on an owner, a snapshot on a viewer, or ``None``."""
        ...

    @property
    def mcp_startup(self) -> Any | None:
        """The MCP startup outcome (discovery failures), or ``None``."""
        ...

    # The registries behind ``/agent``, ``/team`` and ``/project``. Each is the
    # registry of the machine the VIEWER runs on, which is why a viewer is a
    # real implementation and not a passthrough.

    @property
    def agent_registry(self) -> Any | None:
        """The user's agent-profile registry, or ``None`` when none is wired."""
        ...

    @property
    def team_registry(self) -> Any | None:
        """The user's team registry, or ``None`` when none is wired."""
        ...

    @property
    def project_registry(self) -> Any | None:
        """The user's project registry, or ``None`` when none is wired.

        Under the same rule as its two neighbours: ``/project`` names projects
        on the machine the VIEWER runs on, so a viewer implements it lazily
        off its own config dir rather than proxying the owner's.
        """
        ...

    # Per-frame reads with a narrow accessor, so a frame does not pay the
    # whole-state clone ``frontend_state`` costs.

    @property
    def pending_gate(self) -> Any:
        """The parked approval gate, read without the whole-state clone."""
        ...

    @property
    def epoch(self) -> str:
        """The runtime epoch, read without the whole-state clone."""
        ...

    @property
    def has_running_job(self) -> bool:
        """Whether any child is still running, read without the whole-state clone.

        The retention predicate (``SessionInteraction.retained_for_auto_work``)
        asks this as a boolean on every canonical delta of every leased source,
        and it used to answer through ``frontend_state`` — a full deep copy of
        canonical state for one boolean — which is why it belongs beside
        ``pending_gate`` and ``epoch`` in this section rather than with the
        roster-returning members.
        """
        ...

    def frontend_revision(self) -> Any:
        """A token that moves whenever the roster, todos or wakes move.

        Clone-free like the members above: the TUI's dock band and todo panel
        compare it to skip re-deriving a view nothing moved under, which on a
        loaded roster was a whole-state clone and a 252-row re-projection per
        canonical delta. Typed ``Any`` here because the concrete
        ``FrontendRevision`` lives with the store; callers compare it for
        equality only (``FrontendStateStore.revision``).
        """
        ...

    def subscribe_frontend(self, handler: Callable[[Any], Any]) -> Any:
        """Refresh, snapshot and subscribe to canonical state in one step.

        Only the argument every session can honour is declared. The owner
        version also accepts ``display_window`` (it can capture a signed page
        of its own transcript); a host that needs that holds the owner type.
        """
        ...

    # The status band and the transcript. ``record_shell`` is routed by a
    # viewer and persisted by the owner; ``restored_usage`` is the provider's
    # own last reading, so the band can seed a truthful zero-cost state.

    def context_breakdown(self) -> dict[str, int]:
        """On-demand token breakdown for the context the next request sends."""
        ...

    def restored_usage(self) -> Usage | None:
        """The provider's own last usage reading for this conversation."""
        ...

    def restored_search_spend(self) -> tuple[dict[str, Any], ...]:
        """Search-spend rows this conversation's transcript carries, oldest first.

        The search twin of :meth:`restored_usage`, and DECLARED for the same
        reason: the TUI reads it off the session to seed a resumed
        conversation's ledger, and a duck-typed ``getattr`` would make a rename
        degrade the band to a silently-short figure instead of an error.
        """
        ...

    def restored_spend(self) -> SessionSpend | None:
        """The durable per-session spend this conversation carries, or ``None``.

        DECLARED on the SHARED protocol rather than on
        :class:`ViewerSessionProtocol`, and the distinction is the one that rule
        is for: ``tui/app.py::_restore_reported_usage`` reads it through a
        duck-typed binding that may hold EITHER kind of session (the band is
        restored on adopt, for an owner runtime and for an attached facade
        alike), and BOTH classes implement it -- the owner by recalling the
        ``session_spend.v1`` row it writes, a viewer by recalling the same row
        out of the journal suffix it already read on a cold open. Declaring it
        on the viewer-only protocol would say a runtime lacks the member, which
        is the opposite of the truth, and would let the owner-side read degrade
        to a silent ``None`` (an unmarked total on screen) on a rename.
        """
        ...

    async def record_shell(self, command: str, result: ToolResult) -> None:
        """Persist a user-typed bang-mode command into the conversation."""
        ...

    # The capability-style members. Each is a REAL operation on both classes,
    # and each host reads it through ``getattr`` + a ``callable`` check even
    # now; declaring them is what turns "the double quietly lacks it" into a
    # static failure, which is the entire point of this block.

    async def fork_snapshot(self, message: str = "") -> dict[str, Any]:
        """Fork the committed prefix without interrupting the live loop."""
        ...

    async def refresh_attention(self) -> dict[str, Any]:
        """Reconcile cross-process attention receipts."""
        ...

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        """Acknowledge one observed attention outcome."""
        ...

    @property
    def active_agent(self) -> str:
        """Display name of the ``/agent`` profile in force (``""`` when none)."""
        ...

    @property
    def active_team_name(self) -> str:
        """Name of the team this session manages (``""`` when none)."""
        ...


@runtime_checkable
class GoalRecordProtocol(Protocol):
    """The judged-goal RECORD's owner-side surface — the ``Session``'s, not a viewer's.

    Why this is neither :class:`SessionProtocol` nor :class:`ViewerSessionProtocol`,
    which is the only interesting question about it:

    * Not on ``SessionProtocol``, because four of these members MUTATE the record
      (``arm_goal``, ``mark_goal_done``, ``delete_goal``, ``dismiss_goal``) and a
      follower must never write the owner's goal — that is the second-writer
      class the ownership rule (``goal_judge.owns_the_session``) exists to
      prevent. A protocol is a promise, and promising every ``SessionProtocol``
      implementation the right to settle someone else's goal would be a false
      one. ``isinstance`` on a ``runtime_checkable`` protocol checks member
      PRESENCE, so declaring them here would also make the facade claim to have
      them when it does not.
    * Not on ``ViewerSessionProtocol``, because the TUI reaches these through
      ``self._session``, which may hold either kind — and pyright would then
      still (correctly) refuse the reads on the owner's own session.

    So it is a THIRD, owner-side protocol, and a host narrows to it deliberately:
    ``isinstance(session, GoalRecordProtocol)`` is False on an attached facade,
    which is the truthful answer to "is this session's goal mine to write?".

    **A follower is not left without the acts.** It reaches all four by the
    ROUTED slash command — ``/goal`` is not in ``frontend_state``'s
    ``_FRONTEND_LOCAL_SLASHES``, so the follower's ``/goal --clear``/``--done``/
    ``--dismiss`` runs on the owner and comes back as the owner's own receipt
    (``tui/app.py::_goal_slash_result``). Adding routed implementations of these
    four to ``AttachedSession`` would be a SECOND path to the same acts beside
    that established one, which is why the facade is left alone here.

    What it covers and why: the five members the TUI calls DIRECTLY (pyright
    reported twelve attribute errors for them against ``SessionProtocol``), plus
    the record's read side and the judge's journal writer, which the TUI reaches
    through delibate ``getattr`` probes with follower-safe defaults. Those probes
    stay probes — a viewer rendering an empty card for a record it cannot see is
    the honest outcome — but the names they spell must still be declared
    somewhere, or a rename degrades to a silent ``None``
    (``tests/unit/session/test_viewer_protocol.py`` is the guard that says so,
    and it knows this protocol by name).
    """

    @property
    def goal_status(self) -> str:
        """The standing goal's lifecycle state: ``"" | "active" | "done"``."""
        ...

    @property
    def goal_judge(self) -> dict[str, Any] | None:
        """The live judge state in its wire shape, or ``None`` with no goal."""
        ...

    @property
    def goal_history(self) -> list[dict[str, Any]]:
        """Settled goals, newest first, as wire dicts."""
        ...

    @property
    def goal_turn_serial(self) -> int:
        """The monotone turn counter the judge samples at a turn end."""
        ...

    @property
    def goal_judge_state(self) -> "GoalJudgeState":
        """The judge's RECORD, not the wire dict — what the driver reads.

        Distinct from :attr:`goal_judge` on purpose: the wire shape drops the
        breaker's ``failures`` counter, and the driver needs it to honour the
        breaker across a restart. Declared because BOTH owning hosts read it off
        the session binding, and because a rename that silently returned
        ``None`` here would stop the judge re-arming rather than failing.
        """
        ...

    @property
    def goal_token(self) -> str:
        """The goal's identity — minted when it is armed, empty when unset.

        Declared here rather than probed because the judge captures it once per
        iteration and drops a verdict the goal has since moved away from; a
        rename that silently returned ``""`` would disarm every judge run instead
        of failing.
        """
        ...

    def arm_goal(self, text: str) -> str:
        """``/goal <text>``: store the objective, mark it active, arm the judge."""
        ...

    def mark_goal_done(self, reason: str = "") -> "GoalHistoryEntry | None":
        """Settle the standing goal as DONE and record it; ``None`` if there was none."""
        ...

    def delete_goal(self) -> str:
        """``/goal --clear``: delete the goal, record nothing, return what went."""
        ...

    def dismiss_goal(self) -> bool:
        """Drop the done chip; ``False`` when there is nothing to dismiss."""
        ...

    def history_view(self, limit: int | None = None) -> list[dict[str, Any]]:
        """``/goal --history``'s payload: settled goals, newest first."""
        ...

    def note_goal_judge(
        self,
        *,
        state: str | None = None,
        run: int | None = None,
        verdict: str | None = None,
        reason: str | None = None,
        failures: int | None = None,
    ) -> None:
        """Journal one judge transition and publish it — the judge's only writer."""
        ...
