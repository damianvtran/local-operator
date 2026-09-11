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

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Deferred: ``frontend_state`` imports ``tui.costs``, whose package
    # ``__init__`` imports THIS module, so a runtime import here is a cycle.
    # The annotation is all that is needed — ``from __future__ import
    # annotations`` above makes every annotation in this file a string, and a
    # Protocol member's type is never evaluated at runtime.
    from local_operator.session.frontend_state import FrontendSessionState


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
#: socket); ``"unknown"`` is the conservative arm for a facade that cannot
#: prove either — it exists because the honest answer to "is a config write
#: going to govern this runtime?" is sometimes "cannot tell", and collapsing
#: that into a bool is what made the predicate it replaces wrong five times.
#:
#: There is deliberately no ``"another-machine"`` member. Every listener and
#: every dialer in this tree binds or dials ``127.0.0.1`` only
#: (``session/runtime/server.py``, ``viewer_server.py``, ``control.py``,
#: ``viewer_client.py``, ``mobile/attach_client.py``, ``peer_client.py``,
#: ``daemon.py``), and ``viewer_server.py`` states it as a security invariant.
#: A cross-host runtime cannot occur, so naming it would re-create the dead
#: axis ``is_remote`` named.
RuntimeLocality = Literal["this-process", "this-machine", "unknown"]


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
    than stylistic. This protocol carries 84 public members and a POSITIVE
    ``isinstance`` walks every one of them; measured on an arm64 host, CPython
    3.12.13, min-of-seven over 2,000 iterations:

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

    async def attach_existing(self) -> bool:
        """Bind to a runtime if one is already live, without starting one.

        The desktop host calls this on a cold viewer so that serving a history
        read never promotes a reader into an executor: losing the runtime must
        not move execution into the HTTP worker.
        """
        ...

    async def update_desktop_watch(self, *, visible: bool, can_notify: bool) -> None:
        """Renew this viewer's desktop attach lease.

        Desktop-surface viewers only; the host recomputes visibility and
        notifiability from its live subscribers and pushes the result so a
        bare proxy socket is not mistaken for a watching human.
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

    def move_will_wait(self) -> bool:
        """Whether a move would block on an in-flight turn."""
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

    def set_cancel_resolution(self, resolver: Callable[[int], None] | None) -> None:
        """Arm the seam that rewrites an optimistic cancel count.

        A follower's Esc reports the count it can see synchronously; the
        authoritative number resolves on the runtime and arrives later.
        """
        ...

    def set_recall_resolution(self, resolver: Callable[[str], None] | None) -> None:
        """The recall twin of :meth:`set_cancel_resolution`."""
        ...

    # --- engine state a viewer host renders --------------------------------
    #
    # The MCP status segment and its menus, the subagent and job views, the
    # sidebar's gate identity, the wake panel, the ``/agent`` and ``/team``
    # listings, and the saved-usage band. These live on THIS protocol rather
    # than on ``SessionProtocol`` because that is the honest claim. BOTH
    # classes implement them — the owner exposes the live manager, scheduler or
    # registry, and a viewer a read-only SNAPSHOT the runtime published — but
    # every host that reads them (the TUI, the desktop routes,
    # ``info/collect.py``) holds an attached facade, so the viewer is where a
    # reader actually lands.
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
    # ``SessionProtocol`` put the whole ``tests/unit/tui`` double population off
    # conformance (measured: 213 pyright errors across 19 files, from ~10 local
    # ``FakeSession`` classes alone; 1996 when every member of the old exclusion
    # list went there). A double that has to grow an MCP manager to keep
    # compiling is a double describing an object the hosts never read. The
    # viewer protocol is the narrower, true claim and costs the doubles nothing.

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

    # The registries behind ``/agent`` and ``/team``. Each is the registry of
    # the machine the VIEWER runs on, which is why a viewer is a real
    # implementation and not a passthrough.

    @property
    def agent_registry(self) -> Any | None:
        """The user's agent-profile registry, or ``None`` when none is wired."""
        ...

    @property
    def team_registry(self) -> Any | None:
        """The user's team registry, or ``None`` when none is wired."""
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
