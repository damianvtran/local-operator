"""Additive HTTP response models over the canonical runtime's own state schema.

Keep FrontendSync and SlashResult shared with attach clients. A parallel HTTP
projection would drop new runtime fields and turn unknown accounting into zeros.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.session.frontend_state import FrontendSync, SlashResult


class SessionRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str
    mtime: float
    #: The session's most recent assistant reply, condensed for a list row, or
    #: ``""`` when it has none yet.
    #:
    #: Read from the canonical transcript, which is where a canonical session's
    #: conversation actually lives. A conversation list that rendered the legacy
    #: agent record's ``last_message`` said "No messages yet" about sessions
    #: holding a full transcript, because nothing on this path ever writes that
    #: field (design D19).
    preview: str = ""


class SessionList(BaseModel):
    """One page of conversations, plus what could NOT be read while building it.

    ``degraded`` names the live-decoration sources whose read failed for this
    page (see ``session.catalog.DECORATION_SOURCES``), and it exists because
    ``active: false`` on a row is ambiguous without it: those fields are
    DEFAULTS when their source could not be read, and a defaulted verdict is
    indistinguishable from a measured one at the client. A renderer that shows
    "nothing is running" over it is asserting a negative the server never
    established, which is how a swallowed failure read to the operator as "all
    my active chats disappeared".

    Empty when everything was read. Additive and defaulted, so a client that
    ignores it behaves byte-for-byte as it does today; the field is always
    PRESENT rather than omitted when empty, so a client can tell "nothing to
    report" from "this server is too old to know".
    """

    sessions: list[SessionRow]
    truncated: bool = False
    limit: int = 100
    degraded: list[str] = Field(default_factory=list)


class SessionSearchRow(BaseModel):
    """One past conversation the search admitted, and WHY it did.

    Deliberately not a :class:`SessionRow`: this is the answer to a SEARCH, so
    it carries the two facts only the search can know and a renderer cannot
    recompute. ``rank`` is the relevance tier the row matched in (0 name, 1 id,
    2 body, 3 soft — see ``local_operator.session.session_search``), decided
    against the body digest index, which the client does not have; the order of
    ``sessions`` already follows it, but sending the tier lets a client merge
    these rows into a list it holds locally without losing the ordering. And
    ``body_match`` says the conversation is why the row surfaced, so a client
    can label it instead of showing a row with no visible reason for being in
    the results.

    ``rank`` IS NOT MEANINGFUL WHEN ``query`` IS EMPTY. An empty query is the
    store listing — every session, newest first — and nothing matched anything,
    so the tier is the constant 0 rather than "this matched on its name". A
    client that merges on ``rank`` must therefore treat an empty ``query`` as
    "no ranking", which is also why the desktop renderer never sends one: its
    search box is a filter, and an empty box is the unfiltered list it already
    has.

    ``name``/``mtime``/``forked`` come along for the same reason the phone's
    payload carries them: a client that has never listed this session (a store
    larger than its own page, a row created since its last poll) can still
    render and open it.
    """

    id: str
    name: str
    mtime: float
    forked: bool = False
    rank: int
    body_match: bool = False


class SessionSearch(BaseModel):
    """A search answer: the matching rows, best first, and the query they
    answer.

    ``query`` is echoed rather than assumed: a client debounces keystrokes, so
    responses arrive out of order, and it must be able to tell which of its
    queries this is the answer to without trusting arrival order.
    """

    sessions: list[SessionSearchRow]
    query: str = ""
    limit: int = 100


class CreatedSession(BaseModel):
    binding: dict[str, str | None] = Field(default_factory=dict)
    session_id: str
    replayed: bool = False


class HistoryEntry(BaseModel):
    id: str
    ts: float
    type: str
    payload: dict[str, Any]


class HistoryPage(BaseModel):
    entries: list[HistoryEntry]
    has_more: bool
    cursor_missing: bool


#: How a child's transcript read ended, when the absence of rows needs naming.
#:
#: ``pending`` and ``gone`` are the two DIFFERENT absences a reader must not
#: conflate (design § 9.1): ``pending`` is a child whose directory exists and
#: whose ``transcript.jsonl`` does not, so a reader offers "nothing yet" and
#: re-probes on the next pulse; ``gone`` is a missing directory, which is
#: final, so a reader stops asking. ``ready`` covers a file that exists even
#: when it holds no rows — an empty page and an unwritten child are not the
#: same fact, and only the filesystem can tell them apart.
ChildTranscriptState = Literal["ready", "pending", "gone"]


class ChildTranscriptPage(HistoryPage):
    """One page of a CHILD's transcript, in the parent's own envelope.

    Derived by the backend and by nothing else (design § 9.1): the two stores
    that know a subagent exists are not witnesses to whether it has written,
    so a renderer inferring ``state`` from a roster row's status would report a
    running child as readable the moment it is registered.
    """

    state: ChildTranscriptState


class SnapshotPayload(BaseModel):
    """The snapshot frame's body: canonical state, the durable page, and WHY.

    ``cold`` is the boolean every renderer already reads. The two fields beside
    it make the answer ACTIONABLE rather than merely honest, and both are
    additive — an older backend omits them, and the documented fallback for a
    reader that predates them is ``cold ? "no-runtime" : null``, which is what
    every client assumed before this existed.

    * ``cold_reason`` — which of the three cases a cold read is: no pid holds
      the session's transcript lease (``no-runtime``); one does and did not
      deliver canonical state (``owner-silent``); or the record is finishing work
      in flight first (``owner-leaving``). ``None`` when the frame is live. A
      TOKEN, not a sentence: the copy belongs to the surface, the same discipline
      the error ladder's ``code`` follows.
    * ``attaching`` — an authenticated dial is retained and its canonical state
      has not arrived yet. The reads keep answering from disk meanwhile, and the
      sync that lands later publishes the rollover the renderer already handles
      for an epoch change.

    Defaulted rather than required so a payload built by a host that does not
    track the distinction (an in-process one, a test's stand-in) still validates:
    ``cold_reason`` defaults to ``None`` and ``attaching`` to ``False``.
    """

    frontend: FrontendSync
    history: HistoryPage
    cold: bool
    cold_reason: Literal["no-runtime", "owner-silent", "owner-leaving"] | None = None
    attaching: bool = False


class DraftPreviewPayload(BaseModel):
    """The strip's readings for a conversation that does not exist yet.

    The SAME canonical projection a cold session publishes (``bridge.state()``),
    so the renderer keeps one arithmetic path and the draft's chips are computed
    by the code that will compute them a moment later for the real session.

    What is deliberately absent is the rest of a snapshot: no ``session_id``
    (there is none — ``frontend.snapshot.session_id`` is empty), no history page
    and no ``cold`` flag. This payload is handed to the status strip ONLY and
    never admitted to the canonical store, so a renderer must not treat it as a
    session it can address.
    """

    frontend: FrontendSync


class SessionSnapshot(BaseModel):
    session_id: str
    epoch: str
    seq: int = Field(ge=0)
    type: Literal["snapshot"]
    payload: SnapshotPayload


#: The three dispositions an admission receipt can report, declared ONCE for
#: both sides of the wire: the model below serialises them, and the route that
#: builds the receipt types its own outcome (and its status constants) with this
#: same alias, so a status the model would reject cannot be constructed in the
#: first place — it is a pyright error at the constant rather than a pydantic
#: failure at response time, on the caller's side of the wire (review round 3,
#: NIT-2). ``MessageAdmission`` deliberately does NOT use it: that route awaits
#: its acknowledgement and can only ever answer ``admitted``.
AdmissionStatus = Literal["admitted", "pending", "failed"]


class AdmissionDetail(BaseModel):
    """What a host reports about one request's admission to the owner.

    ``status`` is the ONE-WORD answer to "did the owner take this text", and it
    is the field a renderer branches on, so it has to be TRUE rather than
    reassuring: a request nobody has accepted is NOT ``admitted``.

    * ``admitted`` — the owner acknowledged the admission. ``detail`` is
      normally the owner's own sentence, passed through verbatim.
    * ``pending`` — the acknowledgement had not arrived when the receipt was
      sent. The request was written to the owner's connection; whether it was
      taken is unknown, and the frame a later FAILURE arrives on
      (``admission.failed``, ``DESKTOP_API.md``) is how the UI learns otherwise.
    * ``failed`` — the owner (or the transport) answered with an error. The
      request was NOT admitted, so the caller may issue a new one.

    A host that CANNOT observe an owner acknowledgement — an in-process one, or
    one whose owner answers synchronously — is free to report only ``admitted``
    (and raise) rather than pretending to a wait it never made: see
    ``MessageAdmission`` below for the ``/messages`` route, which awaits its ack
    in full.
    """

    status: AdmissionStatus
    duplicate: bool
    detail: str


class MessageAdmission(BaseModel):
    """The ``/messages`` route's receipt: ONE disposition, because it waits.

    Deliberately NOT a subclass of :class:`AdmissionDetail`, and the reason is
    the one thing a subclass would get wrong: narrowing a mutable field's
    ``Literal`` in a subclass is not a narrowing at all (pyright's
    ``reportIncompatibleVariableOverride`` says so, and it is right), so the
    inheritance would advertise ``pending``/``failed`` on a route that awaits its
    acknowledgement to completion and can therefore answer neither — it returns
    ``admitted`` or raises the 503 ladder in ``errors()``. A field type here is
    the API contract for the Electron implementers, so it states what this route
    can actually produce.
    """

    status: Literal["admitted"]
    duplicate: bool
    detail: str
    command_id: str
    replayed: bool = False


class OwnerCommandResult(SlashResult):
    admission: AdmissionDetail | None = None


class NativeField(BaseModel):
    name: str
    kind: Literal["text", "secret", "choice", "sessions", "boolean"]
    value: Any = None
    required: bool = False
    choices: list[str] = Field(default_factory=list)


class NativeAction(BaseModel):
    kind: Literal["native_action"]
    destination: str
    session_id: str
    args: str
    fields: list[NativeField] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class CommandReceipt(BaseModel):
    command: str
    result: NativeAction | OwnerCommandResult
    replayed: bool = False


class AnswerReceipt(BaseModel):
    detail: str


class WatchReceipt(BaseModel):
    lease_seconds: Literal[45]


class WarmReceipt(BaseModel):
    """What a speculative engage found, at the moment it answered.

    A STATE, NOT AN OUTCOME. ``warming`` says an engage is under way, and says
    nothing about whether it will succeed — by the time it settles this request
    is long finished. The surface that must report an engage failure is the
    send, which engages again through the same lock and has a user waiting on
    the answer; reporting it twice would put a spawn error in front of someone
    who has so far only typed a character.

    ``cold`` is therefore reserved for "nothing was started", not for "something
    was started and failed", and it rides a 200 like the other two: a warm the
    user did not request must never become an error they have to read.

    ``cold`` IS CURRENTLY UNREACHABLE, and is published anyway. The bridge
    starts a task unconditionally once the viewer is cold and no engage is in
    flight, so today every answer is ``warm`` or ``warming``; whether a runtime
    could actually start (no provider, no model) is a question only the spawn
    itself answers, and it answers it after this response is gone. The member
    stays because it is the honest name for a refusal this route may later
    learn to make cheaply — and because a client that already accepts three
    states costs nothing, while widening the union later would be a contract
    change every renderer has to be taught. Do not narrow it to two.
    """

    state: Literal["warm", "warming", "cold"]


class InterruptReceipt(BaseModel):
    """What an interrupt did, and what the owner said about it.

    THE RUNG BETWEEN "keep going" AND THE KILL SWITCH. ``interrupted`` means
    the owner stopped work that was there and left the session, its runtime and
    its process alive; ``idle`` means there was nothing to stop — a COLD session
    (never engaged, deliberately not spawned to answer this) or one sitting
    between turns. Neither word is decoration: a caller told ``interrupted`` has
    been told something happened, so the route answers ``idle`` whenever
    ``_work_is_running`` says nothing would be stopped rather than reporting a
    press that found an empty session as a press that stopped something.
    ``idle`` rides a 200 like every other honest answer here: a user pressing
    Stop on a session that has already settled has not made a mistake, and
    answering them with an error would put a failure in front of a press that
    succeeded.

    ``receipt`` is the RUNTIME's own sentence, verbatim and never composed
    here. It counts what actually settled and names anything that refused to
    die, which is knowledge only the owner has: a follower that re-worded it
    would be guessing at the number it is refusing to parse. It is ``""`` for
    ``idle`` because there is no owner sentence to report — an invented one
    would be the same class of overstatement the receipt itself is written to
    avoid.

    ``children_running`` and ``background_jobs`` are read off the follower's
    published roster, AFTER the interrupt when there was one, so the surface can
    word its own notice ("2 subagents are still running") without parsing prose.
    They are the follower's view at answer time and can lag the owner by a delta,
    so they are copy inputs and never authority — the receipt is. Split by job
    type because the two have different remaining levers: subagents die with the
    turn, a backgrounded ``bash`` job was deliberately never touched. On an
    ``idle`` answer they are not zeros by default either: nothing was stopped, so
    they describe what IS running, which is how a build that this rung
    deliberately spares is still visible beside an ``idle`` status.
    """

    status: Literal["interrupted", "idle"]
    receipt: str
    children_running: int
    background_jobs: int
    #: Set by the receipt journal when this answer was replayed rather than
    #: re-run (``DesktopReceipts.run``), and declared for the same reason
    #: ``MoveReceipt`` declares it: FastAPI validates the reply against this
    #: model as the route's ``response_model``, so an undeclared ``replayed``
    #: is silently dropped on the way out.
    replayed: bool = False


class MoveReceipt(BaseModel):
    """What a working-directory change did, and where it left the session.

    ``cwd`` is the directory now IN FORCE (absolute, normalised) and ``label``
    is the backend's own home-aware rendering of it. Both are returned because
    they are different facts: the caller compares the absolute path against the
    value the frontend state stream later reports, and prints the label, and
    only the process that owns the session knows how to spell ``~``.
    """

    cwd: str
    label: str
    #: ``set_working_directory``'s own vocabulary -- a COLD viewer is a field
    #: assignment and a BOUND one is a runtime rebind -- plus one route-level
    #: value, ``unchanged``, for a target the session is already in. Deliberately
    #: NOT a new enum to learn elsewhere: the two moved outcomes are the words
    #: the facade already returns, and ``unchanged`` is the receipt the TUI
    #: already prints for the same case ("already in ~/x").
    outcome: Literal["cold", "rebound", "unchanged"]
    #: Whether the move was about to make the user WAIT when it was sampled --
    #: i.e. ``AttachedSession.move_will_wait()`` immediately before the call.
    #: A HINT with the same status the TUI gives it, and deliberately NOT
    #: load-bearing for the UI, which cannot use a pre-call sample to narrate
    #: anything: it learns this value after the move has already happened.
    #: It is here so an operator (and the route's tests) can tell "the session
    #: restarted" from "the field moved under an engage that then failed".
    will_wait: bool
    #: Set by the receipt journal when this answer was replayed rather than
    #: re-run (``DesktopReceipts.run``), exactly as ``CreatedSession`` and
    #: ``CommandReceipt`` do. It MUST be a declared field and not merely a key
    #: the journal adds: FastAPI validates the reply against this model as the
    #: route's ``response_model``, so an undeclared ``replayed`` is silently
    #: dropped on the way out and a replay becomes indistinguishable from a
    #: fresh run. For a move that matters more than for a create: a client that
    #: cannot tell would narrate a restart that did not happen twice.
    replayed: bool = False


class NotificationClaim(BaseModel):
    """Whether THIS surface may raise the banner for one completion.

    ``false`` is an ordinary answer, not an error: another observer on this
    machine (a TUI watching the same session) claimed it first, or the token is
    unknown to the store. Either way the caller's correct behaviour is the
    same — stay quiet — so the distinction is deliberately not reported.

    Says nothing about whether the user READ anything: the claim writes the
    delivery watermark only, and the sidebar's unseen mark survives it.
    """

    claimed: bool


class AttentionState(BaseModel):
    """The shared read watermark, mirrored by the UI's `CompletionAttention`.

    Typed rather than `dict[str, Any]` so the renderer's hand-written copy of
    this shape cannot drift from the authority silently: the field set is the
    contract both sides agree on.
    """

    conversation_id: str
    completion_token: str | None
    anchor_id: str | None
    kind: Literal["complete", "error", "interrupted"] | None
    unseen: bool
    #: ``[published, acknowledged]`` -- monotonic per conversation, and
    #: independent of the runtime epoch, so it orders a conversation's own states
    #: rather than depending on arrival order.
    #:
    #: NOT a merge key: a heal deliberately REPUBLISHES a corrected state under
    #: the SAME pair, so a client that dropped an update whose revision did not
    #: advance would discard exactly the correction the heal exists to deliver.
    #: The bridge republishes on full-state inequality, and no client currently
    #: reads this field; it is a diagnostic ordering hint, and any future
    #: consumer must treat an equal pair as "possibly changed", never as stale.
    revision: list[int]
    #: Absent on the cold list path; only a live runtime can answer it.
    supported: bool | None = None


class PresenceReceipt(BaseModel):
    """The acknowledgment of one delivery-presence beat.

    A receipt rather than a boolean because the client has to know WHEN to beat
    again, and deriving that from a value it hardcoded is how the two sides end
    up disagreeing about the lease after one of them is tuned. The server owns
    the TTL, so the server states it.
    """

    lease_seconds: int


class PresenceKinds(BaseModel):
    """Which notification kinds a connected desktop app claims it can deliver.

    ITS OWN MODEL because it is its own promise, narrower than "the app is
    reachable". The machine-wide feed carries COMPLETIONS ONLY, so a presence
    that claimed every kind would silence a background session's parked `ask`
    with nothing to replace it — the gate cards ride a per-session bridge the
    app holds only for the session it is displaying. Naming the kinds makes that
    limitation part of the wire rather than a property of today's feed.
    """

    can_notify_kinds: list[str]
