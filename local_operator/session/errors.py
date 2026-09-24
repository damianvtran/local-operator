"""Known-safe admission rejections shared by owner, attach client and HTTP.

Only these categories cross the transport boundary as actionable user errors.
Never certify a message by its wording: an arbitrary RuntimeError may contain
socket addresses, credentials or another conversation's identity.
"""

# The two departure phrases, imported rather than retyped, so the sentence this
# module picks and the phrase the runtime publishes cannot drift apart: the
# table below is keyed on them, and a reworded phrase must fail at import time
# rather than silently stop matching. Safe to import here (no cycle):
# ``session.runtime.types`` reaches ``session.retention`` and the stdlib only.
from local_operator.session.runtime.types import (
    LEAVING_FOR_BUILD,
    LEAVING_FOR_BUILD_OVERDUE,
    LEAVING_ON_SIGNAL,
)


class AttachmentUnavailable(ValueError):
    code = "unresolved_attachment"

    def __init__(self) -> None:
        super().__init__(
            "The attached profile or team could not be restored. "
            "Choose an available profile or detach it before sending."
        )


#: The sentence ``Session.prompt`` raises when a turn (or a compaction) already
#: holds the lock. Public because four call sites classify that refusal —
#: ``mobile/attach_client``, ``mobile/tui_handle``, ``session/runtime/serving``
#: and ``session/runtime/process`` — and matching it as text is a seam that
#: breaks silently when the wording changes (agent review round 2, MINOR-1).
TURN_IN_FLIGHT = "session is already streaming; use steer() to inject mid-turn"


class TurnInFlight(RuntimeError):
    """``prompt`` was called while a turn holds the session's lock.

    The TYPED form of :data:`TURN_IN_FLIGHT`, so a caller can decide what to do
    about it (the spooled-owner drain steers the message into the turn in flight)
    instead of pattern-matching a sentence. A ``RuntimeError`` subclass, so every
    existing catcher of the plain raise is unaffected — which is why the three
    call sites outside ``session/`` also accept the text: on a version skew the
    producer may be a build that has never heard of this class, and the old
    sentence is the only seam the two ends share.
    """


class RuntimeRetiring(ValueError, RuntimeError):
    """This runtime has committed to leaving; the message was not admitted.

    A DRAIN, not a failure: the runtime is leaving — for a build handover it is
    handing over to a successor that boots the build now on disk, for a
    termination it is simply finishing what is in flight and going — and it
    refuses new work either way while it finishes what is already in flight. The
    refusal is therefore transient and self-healing, which is what the wording
    and the app's notice register both have to say.

    WHICH DEPARTURE IS THE SENTENCE'S OWN HALF, so it is a parameter rather than
    a constant: the two do not describe the same thing, and only one of them has
    a successor coming (design round 4, D10; agent review round 4, MAJOR-2). See
    ``HEAD_SIGNALLED``.

    The sentence is rebuilt HERE rather than crossing the wire, so the category
    and its copy cannot drift, and an older peer that does not know the code
    still receives it as the frame's ``message``. It is also the only thing a
    user reads about this whole mechanism, which is what makes the vocabulary
    the contract: the owner's previous wording named an internal log token
    (``runtime-retired``) and described the machinery ("the next engage runs
    the new build") rather than the situation the operator is now in (design
    round 1, D2/D3; UX round 1, U3).

    It says NOTHING about where the draft is, and that is load-bearing rather
    than modest: the same category is reached from the peer-send spool fallback
    (a sender whose message could not be spooled), where there is no composer at
    all. The viewer appends that claim where it IS true, the same way it does
    for the oversize refusal one branch over.

    BOTH BASES, deliberately. ``admission_error`` decodes this family as
    ``ValueError`` (its other two categories are), and the refusal has been
    raised as a bare ``RuntimeError`` since it existed — by the runtime's own
    admission gates, by the peer-send spool fallback, and by every test that
    pins them. A ``ValueError``-only class would have silently changed the catch
    shape of a refusal three call sites already handle as ``RuntimeError``, the
    same trap ``OwnerAckTimeout(ConnectionError, TimeoutError)`` records one
    module over.
    """

    code = "runtime_retiring"

    #: The departures this refusal can describe, as the two enumerated values
    #: that cross the transport (``error_trigger``). Enumerated for the reason
    #: the module docstring gives for the codes: the far side rebuilds the
    #: SENTENCE from a category, so the only thing that may ride along is a token
    #: from a closed set, never text this side composed.
    #:
    #: WHICH OF THE TWO A RAISER ACTUALLY SENDS, because a reader tracing the
    #: field should not have to go hunting for a producer that is not there:
    #: ``SIGNAL`` is the one, raised by
    #: ``serving.ServingSessionHandle._retiring_refusal`` from the cause its own
    #: latch committed (``types.SIGNAL_DRAIN_CAUSE``). ``BUILD`` is raised by the
    #: runtime whose drain is still running — ``ServingSessionHandle.
    #: _retiring_refusal`` names it from ``_draining`` without a committed exit,
    #: which is the term that tells a build drain from the ``/move`` retirement
    #: sharing its cause — and it is also what the far side resolves a build
    #: drain TO from the phrase that drain published.
    SIGNAL = "signal"
    BUILD = "build"

    #: The sentence, in the halves a viewer needs. ``HEAD`` states the situation,
    #: ``TAIL`` names the one act left; the owner's own rendering keeps them
    #: joined by ``REFUSED``, and a viewer with a COMPOSER inserts its claim
    #: between them instead of bolting it on after the full stop. Measured on the
    #: appended form (design round 3, D1; UX round 3, U5): two em dashes in one
    #: paragraph, a fragment opening after a ``.``, a strand at 100 columns and a
    #: one-word last line (``composer``) at 60 — in the one sentence whose job is
    #: to say the operator's work is safe. Exposed rather than re-composed so the
    #: two ends cannot drift.
    HEAD = "This session is switching to a newer build; the one it loaded is gone from disk."
    #: The same half for the OTHER departure that reaches this refusal.
    #:
    #: A REFUSAL IS ABOUT A DEPARTURE, and the departure is not always a build:
    #: ``ServingSessionHandle.prompt`` refuses from ``begin_drain``, which the
    #: SIGTERM path latches too, so a signalled runtime refused a message with
    #: "the one it loaded is gone from disk" — a build that does not exist and is
    #: not coming, painted under a notice that correctly said the session had
    #: been signalled to stop (design round 4, D10; agent review round 4,
    #: MAJOR-2). It keeps the situation clause the signal NOTICE uses so the two
    #: rows read as one event, and drops every build claim, exactly as the signal
    #: notice does.
    HEAD_SIGNALLED = "This session was signalled to stop; it will not start a new turn."
    #: The sentence for a departure NOBODY NAMED, and it names none either.
    #:
    #: Reached only when the raiser sent no token (every build older than the
    #: field) AND the far side's own phrase established no trigger — a frame that
    #: named neither, or a refusal whose connection saw no draining frame at all.
    #: The old answer was the build sentence, which is what round 5 filed: with a
    #: signal-draining runtime from this branch's own older builds (the pre-key
    #: rungs of PR #1141, e.g. `8dd605365`) the viewer painted "switching to a
    #: newer build; the one it loaded is gone from disk" directly under a notice
    #: that said the session had been signalled to stop — the contradiction this
    #: PR exists to remove (agent review round 5, MINOR-1; UX round 5, U14; design
    #: round 5, D11). Correct for a RELEASED build, whose only draining announce is
    #: the stale-build handover; false for those.
    #:
    #: WHY IT SAYS ONLY THIS. The instance is built only by a latched departure, so
    #: "leaving, and it will not start a turn" is the one thing this gate
    #: establishes by itself — true of both unnamed raisers, while anything
    #: narrower is not. In particular the drain notice's neutral sentence ("it is
    #: finishing in-flight work first") is not borrowed here: that clause is
    #: established by a frame that said ``draining``, and one unnamed raiser — the
    #: reaper's ``idle-exit`` latch, which owes no successor and has no work in
    #: flight — never sent one.
    HEAD_UNNAMED = "This session is leaving; it will not start a new turn."
    REFUSED = "The message was not admitted"
    TAIL = "send it again once the session is running again."
    #: The tail for the departure that OWES A SUCCESSOR, reached on the refusal
    #: paths: there was nowhere to spool the message (an unwritable inbox, an
    #: attachment an inbox row cannot carry) or the exit was already committed.
    #: The old tail — "send it again once the session is running again" — sent
    #: the operator to perform the one operation the refusal had just refused,
    #: and it is what the incident left them with for 1 h 40 m (memo §4.2 piece
    #: 3). The old instruction stays; the DESTINATION is what changes.
    #:
    #: IT NAMES NO CARRIAGE, and that is a correction rather than a style
    #: choice: this arm is exactly the one where the message was NOT carried
    #: (QA round 1, Q-2 — the earlier "a newer build is starting here to carry
    #: on" read as "your message is on its way" while the spool had failed, and
    #: a front end that does not restore a draft would never re-send). Asking
    #: for the re-send is the honest instruction here, and naming the build is
    #: what makes it actionable (UX round 1, U5).
    TAIL_HANDOVER = "send it again once the new build is up."
    #: The tail for a message that IS carried — on the successor's spool — and
    #: reaching a caller that cannot watch the turn it will run in: a loop's
    #: ``prompt_and_wait`` correlates on an ``AgentEndEvent`` from THIS runtime,
    #: and the successor writes that row after this process has exited. So the
    #: refusal is the answer, and this is its honest tail: the deferral, named
    #: as a deferral. Never sent over the wire — the spool receipt is what the
    #: runtime answers with — so no ``error_*`` field carries it.
    TAIL_QUEUED = "your message is queued and will run as soon as this session runs again."
    #: The queued tail for a departure that owes NO successor: a signalled stop, or
    #: a phrase this build cannot place. The row IS durable — the next runtime to
    #: open the session drains it — but whether one ever does is the host's
    #: decision, not the signal's, so the sentence must not promise a future the
    #: departure does not establish (the D6 rule the neighbouring notice is split
    #: by; agent review round 2, NIT-1). Hence the conditional.
    TAIL_QUEUED_OTHER = "your message is queued and will run if this session runs again."

    def __init__(self, trigger: str = "", leaving: str = "", *, queued: bool = False) -> None:
        # ``HEAD`` is per-INSTANCE because the situation is: the same refusal
        # carries different sentences for the departures, and the far side
        # rebuilds whichever one the raiser's enumerated ``trigger`` names.
        #
        # A TRIGGER NOBODY NAMED IS NOT ASSUMED TO BE THE BUILD. It used to be,
        # and that fallback claimed more than it had measured: it asserted that a
        # raiser which cannot name a departure is "a runtime older than the field,
        # whose only drain IS the build handover". True of a released runtime
        # (its only draining announce is the stale-build handover) and false of
        # this branch's own intermediate builds, which announce a SIGNAL drain
        # with no token, so the sentence contradicted the notice above it in the
        # one window this PR is about.
        #
        # SO THE FAR SIDE SUPPLIES WHAT THE RAISER COULD NOT, and ``leaving`` is
        # that evidence: the phrase the viewer already derived from the frame it
        # is watching (``types.drain_phrase_for_frame``), which for the pre-key
        # builds is the trigger's own words off the wire, and which is the ONLY
        # thing that tells a build drain from the ``/move`` retirement sharing its
        # cause. It is matched against the two known phrases rather than
        # interpolated — a peer's words may key a table, for the reason ``count``
        # may not be a sentence — and it is read ONLY when the token names nothing,
        # because an explicit token is the raiser's own enumeration of its own
        # latch and outranks an inference. What is left when neither establishes
        # anything is the sentence that names no departure at all.
        self.trigger = trigger if trigger in (self.SIGNAL, self.BUILD) else ""
        if not self.trigger:
            self.trigger = _TRIGGER_FOR_LEAVING.get(leaving, "")
        self.HEAD = _HEADS.get(self.trigger, self.HEAD_UNNAMED)
        # The tail is chosen off the SAME token as the head, for the reason
        # ``_HEADS`` gives below: one departure, one reading of it. A trigger
        # that names nothing keeps the tail this class has always carried.
        #
        # ``queued`` outranks the token because it is a fact about THIS MESSAGE
        # rather than about the departure, and the two sentences are not
        # interchangeable: one asks for a re-send, the other says the message is
        # already on its way.
        self.TAIL = (
            _QUEUED_TAILS.get(self.trigger, self.TAIL_QUEUED_OTHER)
            if queued
            else _TAILS.get(self.trigger, self.TAIL)
        )
        super().__init__(f"{self.HEAD} {self.REFUSED} — {self.TAIL}")


#: Which sentence each enumerated departure earns. Keyed by the token, so the
#: two arms cannot be swapped by editing one branch of an ``if`` — the same shape
#: the app uses to pick its drain NOTICE (``app._DRAIN_NOTICES``), for the same
#: reason: these are two readings of one state and they must be chosen the same
#: way at both ends.
_HEADS: dict[str, str] = {
    RuntimeRetiring.SIGNAL: RuntimeRetiring.HEAD_SIGNALLED,
    RuntimeRetiring.BUILD: RuntimeRetiring.HEAD,
}

#: The tail each enumerated departure earns, chosen off the same token as
#: ``_HEADS`` and for the same reason: they are two readings of one state, and
#: the two ends must pick them the same way. A SIGNAL drain is leaving for good
#: and owes nobody; a BUILD drain is handing the session to the build on disk,
#: which is the one departure where re-sending is not the operator's job.
#: The queued tail per trigger, on the same reasoning as ``_TAILS``: a sentence
#: may state only what the departure establishes, and only a build drain
#: establishes that a successor is coming.
_QUEUED_TAILS: dict[str, str] = {
    RuntimeRetiring.BUILD: RuntimeRetiring.TAIL_QUEUED,
}

_TAILS: dict[str, str] = {
    RuntimeRetiring.BUILD: RuntimeRetiring.TAIL_HANDOVER,
}

#: The departure a phrase establishes, for a raiser that could not name one.
#: Only the phrases the runtime publishes are keys: anything else — an empty
#: phrase, or a phrase written by a build this one has never heard of — is
#: evidence about nothing, and the unnamed sentence is the answer for it.
#:
#: THE BOUNDED HANDOVER RESOLVES TO ``BUILD`` (agent review round 1, N1). It is a
#: build departure in every clause the build head states — the install on disk
#: moved under this runtime, it is leaving for the newer build, and the successor
#: that answers the refusal is that build — so the alternative, no token at all,
#: gave the MOST serious departure the VAGUEST sentence ("This session is
#: leaving…") while an ordinary handover named the build. What the head does not
#: say is that the turn was cut rather than finished; that fact is the phrase's
#: (``types.LEAVING_FOR_BUILD_OVERDUE``, which this table is keyed by and the
#: refusal's receipt quotes) and the record's, and the refusal sentence holds only
#: the token its own category enumerates — a third trigger value would be a wire
#: change to say it twice.
_TRIGGER_FOR_LEAVING: dict[str, str] = {
    LEAVING_ON_SIGNAL: RuntimeRetiring.SIGNAL,
    LEAVING_FOR_BUILD: RuntimeRetiring.BUILD,
    LEAVING_FOR_BUILD_OVERDUE: RuntimeRetiring.BUILD,
}


class OperatorAuthorityRequired(ValueError, RuntimeError):
    """This request would loosen a running gate, and did not come from its console.

    `/approvals auto` and an APPROVED card remove the approval gate that
    constrains the caller, so the runtime additionally demands a per-connection
    proof of the capability its own spawner minted (issue #1310, ``harness/
    approval``). A request that arrives without it — a follower pane, the phone
    relay for a runtime another process started, the desktop app for a session
    its backend did not engage, or a model-authored tool call that merely read
    the session record — is refused with this.

    A TYPED refusal so every route can carry it verbatim. Before this, the
    refusal crossed the socket as an anonymous `error` frame and then a bare
    `RuntimeError`, and each route guessed: the desktop command surface answered
    `503 runtime_unreachable` ("reconnect and reconcile") and the desktop card
    route answered `409 "no longer pending"` while the card was still parked,
    both of which describe a different problem than the one the operator has
    (agent review round 1 R1-2 = design D1 = UX U4 = QA Q1).

    BOTH BASES, deliberately, for the reason ``RuntimeRetiring`` records one
    class down: the routes that carry a control request catch `ValueError` (the
    relay's HTTP arm) or `RuntimeError` (the card route's answer path), and a
    class that satisfied only one would silently change the catch shape of a
    call site that already handles it.

    The message is built HERE from the constant the runtime also sends, so the
    category and its copy cannot drift (the decode path in
    :func:`admission_error` takes no text off the wire).
    """

    code = "operator_authority_required"

    #: Whether this refusal is the UNCONFIGURED variant — the host has no usable
    #: anchor, so NEITHER named remedy can work until `lop operator install` has
    #: run there (UX round 6, U1/U2 — the two halves of one gap). It
    #: selects a different sentence, exactly as ``trigger`` does, and it crosses
    #: the transport as its own CODE rather than as a field, for the reason the
    #: module docstring gives for codes: the far side rebuilds the sentence
    #: locally, so only an enumerated value may ride along.
    unconfigured = False

    #: The op a refusal came from, as one of the enumerated control ops that can
    #: carry an increasing request. ``""`` means the raiser did not say, which
    #: rebuilds the command's sentence — the pre-trigger behaviour.
    CARD_OPS = frozenset({"approval_answer"})

    def __init__(self, message: str | None = None, *, trigger: str = "") -> None:
        # Kept so the transport can forward the TOKEN rather than any prose, and
        # so the far side picks the same sentence locally.
        self.trigger = trigger if trigger in ("slash", "slash_result", "approval_answer") else ""
        if message is None:
            from local_operator.harness.approval import (
                CARD_APPROVAL_REFUSED_NOTICE,
                CARD_APPROVAL_REFUSED_UNCONFIGURED_NOTICE,
                OPERATOR_AUTHORITY_REQUIRED_NOTICE,
                OPERATOR_AUTHORITY_REQUIRED_UNCONFIGURED_NOTICE,
            )

            if self.trigger in self.CARD_OPS:
                message = (
                    CARD_APPROVAL_REFUSED_UNCONFIGURED_NOTICE
                    if self.unconfigured
                    else CARD_APPROVAL_REFUSED_NOTICE
                )
            else:
                message = (
                    OPERATOR_AUTHORITY_REQUIRED_UNCONFIGURED_NOTICE
                    if self.unconfigured
                    else OPERATOR_AUTHORITY_REQUIRED_NOTICE
                )
        super().__init__(message)


class OperatorAuthorityUnconfigured(OperatorAuthorityRequired):
    """The same refusal on a host where no anchor is USABLE, so the remedies differ.

    WHY THIS IS A DIFFERENT CATEGORY RATHER THAN DIFFERENT PROSE. The default state
    of a fresh host between ``lop operator init`` (which only STAGES the anchor) and
    ``lop operator install`` (the privileged step that lands it) is: a correctly
    paired phone signs, and the runtime refuses it — the anchor it would verify
    against is present-but-untrusted or absent. The old sentence named "this
    machine (Touch ID) or your paired phone", and on that host NEITHER can work,
    while the one command that unlocks both was named nowhere (UX round 6, U1/U2).

    A subclass rather than a flag on the message so the code, the level and the
    sentence stay one decision: every route that keys on
    ``operator_authority_required`` keeps working for the ordinary case (it is the
    base class), and the phone can key on this code to say what is actually left.
    """

    code = "operator_authority_unconfigured"
    unconfigured = True


class ProfileRegistryUnavailable(ValueError):
    code = "profile_registry_unavailable"

    def __init__(self, count: int | None = None) -> None:
        """Report HOW MANY definitions are unreadable, never WHICH.

        The offending paths are logged at warning by the raising site. They are
        deliberately not interpolated here: this message crosses the transport
        boundary (see the module docstring), and a path names the operator's
        home directory, while a basename is an agent id. A count is
        content-free but still actionable -- it tells the user whether to look
        for one bad definition or several, and confirms the number is not zero,
        which is what the original wording could not do when the true cause was
        a directory that is not an agent at all.

        ``count`` is optional because the raising site does not always have a
        number: a scan that died on an ``OSError`` never attributed the failure
        to specific directories, and an older peer sends no count at all.
        """
        # Kept so the transport can forward the integer itself rather than
        # re-parsing it out of the rendered sentence.
        self.count = count
        detail = ""
        if count is not None:
            noun = "definition" if count == 1 else "definitions"
            detail = f" {count} agent {noun} could not be read."
        super().__init__(
            "The agent registry could not be read completely." + detail + " "
            "Repair unreadable or invalid agent definitions, then retry. "
            "No packaged profile was substituted."
        )


class AsideUnanswered(ValueError):
    """An off-record aside ended without a text answer.

    The typed home for a failure the shared primitive
    (:meth:`Session.complete_aside`) used to swallow: a bare second tool call
    returned ``""``, which surfaced in the UI as a provider fault rather than as
    the thing that actually happened. An aside carries no tools it may run, so a
    model that answers with a call and then repeats that answer — or with a call
    and then NOTHING AT ALL on the corrected retry — has produced no answer, and
    saying so is the honest outcome. The sentence below is worded for BOTH arms,
    which is not a nicety: the second one is the state a real provider reached
    when QA reproduced this (a tool call once, then silence), and the sentence it
    used to carry — "a tool call … both times it was asked" — was false about
    what happened. The claimed cause must be one the user can recognise in what
    they saw, since it is the only account of the failure they get.

    A ``ValueError`` because that is what :func:`admission_error` decodes and
    what the desktop route ladder's named-refusal arm catches — the same shape
    :class:`AttachmentUnavailable` and :class:`ProfileRegistryUnavailable` use,
    so this rides the existing 409 ``{"code", "message"}`` body rather than
    falling through to a bare 500.

    The sentence is built HERE rather than crossing the wire, for the reason the
    module docstring gives: the far side rebuilds it from :data:`code`, so no
    owner prose — and no provider body — is ever rendered as an operator-facing
    sentence. What a caller should DO is ask again: the retry inside the aside
    is already spent, and a second attempt is the only remaining remedy.

    The goal-loop judge reaches this through the same primitive, and a raise is
    CORRECT there: ``GoalLoop.run`` counts an exception as a judge failure
    (``MAX_LOOP_JUDGE_FAILURES``, ``session/goal_loop.py``), which is the right
    verdict for a judge that cannot answer in text.
    """

    code = "aside_unanswered"

    def __init__(self) -> None:
        super().__init__(
            "The model did not answer your aside in text — either a tool call, "
            "which is not available off the record, or nothing at all. No answer "
            "was produced: ask again."
        )


class AsideEmptyAnswer(AsideUnanswered):
    """A settled aside answer carrying no text — raised at the DESKTOP ROUTE.

    WHY IT IS NOT IN THE PRIMITIVE. ``Session.complete_aside`` returns ``""``
    for a model that answers empty WITHOUT calling a tool, and that contract is
    deliberate: a length-stop or a refusal must reach the goal-loop judge as "no
    verdict" rather than as a fault, and the judge is the primitive's other
    caller. The desktop route has a different obligation because IT is where the
    durable aside entry gets written — a 200 with ``text: ""`` stores an empty
    assistant turn marked complete and adoptable, which the renderer paints as
    no answer and no error, and the panel's next "Ask again" then continues that
    empty exchange instead of starting over. So the refusal belongs at the route,
    where an empty answer can be named for what the user is looking at, and the
    primitive is left exactly as it was.

    A SUBCLASS of :class:`AsideUnanswered` for one answer path, not for the
    sentence: the desktop ladder's named-refusal arm and the asides route's
    drop-on-refusal both key on the base class, so a second unrelated type would
    be a second place to remember. The sentence is restated because the base's
    names a tool call this arm did not involve, and it keeps the same remedy.
    The :data:`code` is its own so a renderer can tell "the model said nothing"
    from "the model tried to use a tool".

    Never crosses the attach wire (it is raised in the daemon that owns the
    route, after the owner returned), so it has no :func:`admission_error` arm:
    the client gets this sentence in the HTTP body, not a code to rebuild.
    """

    code = "aside_empty_answer"

    def __init__(self) -> None:
        ValueError.__init__(
            self,
            "The model answered your aside with no text at all, so the exchange "
            "has no answer to keep. Ask again.",
        )


class MoveIndeterminate(Exception):
    """A move whose owner outcome is UNKNOWN, so nothing may be rolled back.

    WHY THIS IS NOT A ``RuntimeError``. The move route maps ``RuntimeError`` to
    an ordinary 409 refusal and, on that path, the durable marker and the
    viewer's fields are restored — the correct story for a refusal, and the
    WRONG one here. This class is raised when the retire REQUEST reached the
    owner and the answer did not come back definately (a dropped socket, an ack
    timeout): the owner may already have retired and accepted the new
    directory, so restoring the old marker would overwrite a committed move
    with a stale one, and the successor could then spawn in the old path while
    the receipt says the session is there.

    The honest answer is "reconcile before claiming either directory", which is
    what the route turns into a 503 whose body is
    ``{"code": :data:`code`, "message": <the sentence>}`` — the same shape
    ``DaemonRetiring`` and ``SubagentChildUnavailable`` use in that ladder, and
    the shape the desktop client already reads (it takes ``detail.message`` when
    ``detail`` is an object, so a named condition and a plain sentence both
    render). A subsequent move must first finish that reconciliation under the
    per-session move lock rather than act on an optimistic ``_cwd``.

    :attr:`detail` is the underlying cause — transport errno, a marker path, the
    three copies that disagreed — and is deliberately NOT on the wire: it names
    sockets, control ports and directories. Both raise sites LOG it instead,
    because a 503 whose cause is recorded nowhere leaves an operator with a
    generic "reconcile" and no thread to pull: the transport/unknown-outcome
    raise logs the exception with ``exc_info`` where it is still live
    (``session/attached.py``, ``set_working_directory``), and the settlement logs
    all four readbacks plus the path and errno of a repair write that failed
    (``server/utils/desktop_sessions.py``, ``_settle_unconfirmed_move``).

    :attr:`message` is overridden by exactly two callers, and both sentences are
    deliberate. The publication failure
    (``AttachedSession._publish_working_directory``) is its own sentence because
    the move itself IS confirmed there and only the viewer's repaint is not. The
    settlement refusal (``_settle_unconfirmed_move``) is its own because the
    directory is genuinely unresolved — and it names the action (reconnect, then
    reconcile) rather than the transport. Both carry :data:`code`, so a renderer
    keys on the condition instead of on the prose.
    """

    code = "move_outcome_unknown"

    def __init__(self, detail: str = "", *, message: str | None = None) -> None:
        self.detail = detail
        super().__init__(
            message
            or (
                "The move's outcome could not be confirmed. The session may have "
                "moved; reconnect, then reconcile its working directory before "
                "moving again."
            )
        )


class SessionStoreUnavailable(OSError):
    """The session store could not be walked, so no listing built from it is true.

    THE FAILURE THIS EXISTS TO STOP: an unreadable store being reported as an
    EMPTY one. ``resume._scan_sessions`` answered ``[]`` for any ``OSError``
    raised by the ``sessions/`` directory -- ``EMFILE``/``ENFILE`` under file
    descriptor exhaustion, ``EACCES``, ``EIO``, ``ENOTDIR`` -- so the desktop
    list route answered ``200 {"sessions": [], "truncated": false}`` to a
    client that cannot tell that from "you have no conversations". The sidebar
    adopts that answer as MEMBERSHIP and replaces what it is showing, so a
    transient descriptor exhaustion emptied the operator's visible catalogue
    for as long as it lasted, with nothing in the response and nothing at the
    default log level to say why.

    WHY AN ``OSError`` SUBCLASS rather than a plain ``Exception``: every call
    site that already TOLERATES an unreadable store -- the phone daemon's
    search, the CLI's ``/resume`` picker, the retention policy -- tolerates it
    with ``except OSError``, and a parallel hierarchy would silently change
    their catch shape. Subclassing leaves those tolerances exactly as they
    were, while giving the sites that must NOT tolerate it (the catalogue and
    the phone's durable listing, whose answers a UI adopts as membership)
    something typed to catch and map to a retryable sentence instead of an
    empty listing.

    The message names no directory: this one is not echoed to a client -- the
    list route answers with its own vetted 503 sentence, the rule the module
    docstring sets for every category here -- and the cause (which does carry
    the path) rides along as ``__cause__`` for the log.

    THE CODE IS THE POINT OF CARRYING IT, not decoration. The route answers
    this as a 503, and the desktop app puts a ``GET /v1/desktop/sessions`` with
    ``limit=1`` on its identity probe -- the question "is the daemon at this
    address usable with my credential?". A client that can only see the status
    has to read every non-2xx as a refusal, and a transient store blip then
    reads as a capability 403 on a daemon whose credential was never in
    question, which in the app's attach path means declining a live daemon and
    spawning a second one over it. With the code the rule is the cheap one: 401
    and 403 mean the credential was refused, ANY other answered status means a
    daemon answered. Same shape and same reason as ``DaemonRetiring`` and
    ``MoveIndeterminate`` above -- a named retryable condition, not a status a
    caller has to guess from.

    It is inert until a client reads it: the classification is the client's
    half, and it lands with the app (``local-operator-ui``, where only 401/403
    count as "this credential is refused"). What this side owes is the field.

    ``session_store_unavailable`` rather than the shorter ``store_unavailable``
    for the two reasons this family already states one of: the token has to be
    unique in the vocabulary a client keys on, and the MCP credentials tool
    already answers ``store_unavailable`` for a failure to write a SECRET -- a
    different store entirely, and one a client that switched on the bare token
    would be right to try handling the same way. The ``<subsystem>_unavailable``
    spelling is the sibling's (``profile_registry_unavailable``).
    """

    code = "session_store_unavailable"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__("The session store could not be read" + (f": {detail}" if detail else "."))


def admission_error(
    code: str,
    count: int | None = None,
    trigger: str | None = None,
    leaving: str | None = None,
) -> ValueError | None:
    """Decode only an enumerated category, never owner-supplied message text.

    ``count`` is carried as its own integer field rather than being recovered
    from the peer's message, which is the whole point: the wording is rebuilt
    locally from the category, so the only thing crossing the transport is a
    number. An integer cannot name a path, a socket address or another
    conversation's identity, so it does not widen what the module docstring
    admits -- unlike ``str(exc)``, which is why that is still never trusted.

    Anything that is not a plain non-negative ``int`` is dropped rather than
    rendered: the far side is untrusted input, and a caller that omits the
    field (an older runtime) must degrade to the countless wording, not raise.

    ``trigger`` is the same idea for a different kind of value: WHICH departure
    a retirement refusal is about, as one of the enumerated tokens on
    :class:`RuntimeRetiring`. It is validated against those tokens here rather
    than accepted as a string, for the reason the count is: what crosses the
    transport must not be able to carry prose into a sentence this side builds.
    An unknown or missing token means "this raiser cannot name its departure",
    which is the pre-field behaviour.

    ``leaving`` is the ONE argument that is not off the wire: it is what the
    CALLER — the far side, the connection that is watching this runtime — already
    knows about the departure, i.e. the phrase it derived from the ``retiring``
    frame (``types.drain_phrase_for_frame``). It exists for the raiser whose
    build predates ``error_trigger``, which is exactly this branch's own
    intermediate builds: they signal-drain, publish the trigger in the frame's
    own words, and can say nothing in the refusal's fields. It is therefore read
    only where the token names nothing, and it is matched against the two known
    phrases rather than rendered — the frame's phrase is still a peer's words,
    and a table key is all this boundary admits of those (MINOR-1/U14/D11,
    round 5). It never reaches an error object; the trigger it resolves to does.
    """
    if code == AttachmentUnavailable.code:
        return AttachmentUnavailable()
    if code == RuntimeRetiring.code:
        return RuntimeRetiring(
            trigger=trigger if isinstance(trigger, str) else "",
            leaving=leaving if isinstance(leaving, str) else "",
        )
    if code == OperatorAuthorityUnconfigured.code:
        # Ahead of the base class only for legibility: the two codes are
        # distinct strings, so neither can shadow the other.
        return OperatorAuthorityUnconfigured(trigger=trigger if isinstance(trigger, str) else "")
    if code == OperatorAuthorityRequired.code:
        # No PROSE off the wire: the sentence is rebuilt from the constant, and
        # ``trigger`` — one token from a closed set — only chooses which of the
        # two constants that is (a refused command vs a refused card).
        return OperatorAuthorityRequired(trigger=trigger if isinstance(trigger, str) else "")
    if code == ProfileRegistryUnavailable.code:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            count = None
        return ProfileRegistryUnavailable(count=count)
    if code == AsideUnanswered.code:
        # No payload: the sentence is a constant, and there is nothing about the
        # provider's answer this side wants on the wire (it can quote the
        # conversation or a tool name the model invented).
        return AsideUnanswered()
    return None
