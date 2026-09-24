"""Additive HTTP response models over the canonical runtime's own state schema.

Keep FrontendSync and SlashResult shared with attach clients. A parallel HTTP
projection would drop new runtime fields and turn unknown accounting into zeros.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from local_operator.session.frontend_state import FrontendSync, SlashResult
from local_operator.session.runtime.types import reported_subagent_count


class OpenedBy(BaseModel):
    """WHO opened an agent workstream: the frozen ``{agent, label, session}`` object.

    A MODEL rather than ``dict[str, str | None]`` so the three names the desktop
    sidebar (local-operator-ui #448) is written against live in the published
    schema and are ENFORCED at validation: with a free-form dict a producer-side
    rename or an extra key passed silently and surfaced as an empty label in
    another repository (PR #1436 agent review round 1, F4).

    ``extra="forbid"`` is the enforcement half. Every member is REQUIRED-BUT-
    NULLABLE rather than defaulted: the producer (``resume.workstream_opened_by``)
    always writes all three, ``None`` where a member could not be read, so a
    missing key is a producer bug to fail on, not a value to fill. The JSON is
    byte-identical to the dict it replaces — same three keys, same order.
    """

    model_config = ConfigDict(extra="forbid")
    agent: str | None
    label: str | None
    session: str | None


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
    #: Whether the sidebar has this conversation pinned: ALWAYS PRESENT, with
    #: both values, on every row.
    #:
    #: IT MUST NEVER BE OMITTED, and the renderer is why: its row merge is
    #: ``{...current, ...incoming}`` under the rule "an absent key is not a
    #: claim", so a row that arrived without ``pinned`` would leave a stale
    #: optimistic ``true`` in place forever — the pin glyph and the section
    #: membership would outlive a successful unpin made anywhere else. Sending
    #: both values is what makes a list read SETTLE the field.
    #:
    #: REQUIRED rather than defaulted, for that same reason: a projection that
    #: forgot to fill it fails loudly here at validation, instead of publishing
    #: an omitted key whose absence the client is entitled to read as no claim.
    #:
    #: The renderer's ``SessionCatalogueRow`` is this shape's hand-written
    #: mirror, so this key is a change to a second file as well as this one.
    pinned: bool
    #: Whether this conversation is archived: ALWAYS PRESENT, with both values,
    #: on every row — `pinned`'s rule and `pinned`'s reason.
    #:
    #: ARCHIVE AND PIN ARE INDEPENDENT FACTS ABOUT ONE ROW, which is why this is
    #: a second key rather than a state of the first. A conversation can be both
    #: (archived while pinned) and the two mean different things: pinned says
    #: where the row SORTS, archived says whether the row is OFFERED at all. A
    #: renderer that folded them into one field could not represent the case a
    #: user creates in two presses, and would have to invent an answer for it.
    #:
    #: An archived row reaches a client ONLY from a listing that asked for it
    #: (``include_archived=true``). The key is required rather than defaulted for
    #: the reason `pinned` is required: a projection that forgot it would publish
    #: an absence the client is entitled to read as no claim, and an archived
    #: conversation would sit in the ordinary list with nothing saying so.
    archived: bool
    #: How many of this session's OWN delegated children are running, and how
    #: many are parked waiting for a capacity slot — or ``null`` when nothing
    #: reported them.
    #:
    #: DECLARED HERE RATHER THAN LEFT TO ``extra="allow"``. The pair does reach a
    #: client either way — the row is built from ``SessionRow._asdict()`` in
    #: ``server.utils.desktop_sessions`` — but this model IS the contract a client
    #: mirrors by hand (the app's own ``SessionCatalogueRow``), and a count the
    #: contract does not name is one a renderer cannot know to draw, or to leave
    #: alone. ``null`` is the wire's "not reported" and must never be read as
    #: ``0``: zero is a measurement that says there are no children.
    #:
    #: NORMALISED AT THE EDGE, through the one shared rule
    #: (``session.runtime.types.reported_subagent_count``) rather than by a
    #: pydantic coercion. A record is written by whatever process owns it and
    #: ``from_json`` validates nothing, so a damaged value is possible; and here
    #: the difference between degrading and failing is the whole conversation
    #: list, because a validation error on ONE row fails the response. An
    #: unusable value becomes ``null`` — which every client already handles —
    #: instead of a 500 the list cannot survive. The same rule runs in
    #: ``info.collect`` and in the sidebar's ``resume._counted``.
    subagents_running: int | None = None
    subagents_queued: int | None = None

    @field_validator("subagents_running", "subagents_queued", mode="before")
    @classmethod
    def _a_reported_count(cls, value: Any) -> int | None:
        """Refuse an unusable count the way every other reader does."""
        return reported_subagent_count(value)

    #: WHO opened this conversation when an AGENT opened it on the operator's
    #: behalf — ``{"agent": str | None, "label": str | None,
    #: "session": str | None}`` — and ``None`` for every ordinary row.
    #:
    #: ADDITIVE AND FROZEN, for the same reason `degraded` is: the sidebar that
    #: renders the attribution is written against exactly these three names in
    #: another repository, so a rename here would surface as an empty label
    #: rather than as an error anyone sees (``resume.OPENED_BY_KEYS`` owns the
    #: list and the reasoning).
    #:
    #: A NULLABLE OBJECT rather than an omission, which is the opposite of
    #: `pinned`'s rule and deliberately so: ``pinned`` is a state the user
    #: TOGGLES, so an absent key would leave a stale optimistic value standing,
    #: while this is an immutable fact about how the session began and a client
    #: that never renders it is unaffected. ``None`` therefore means "nobody
    #: machine-opened this" and is the answer on every row that is not an agent
    #: workstream.
    #:
    #: Populated ONLY for a workstream row (``resume.ORIGIN_AGENT_WORKSTREAM``).
    #: The 2026-09-18 incident was a machine-started session being
    #: indistinguishable from one the operator opened; a visible workstream
    #: without this would repeat it with the row merely visible instead of
    #: hidden.
    opened_by: OpenedBy | None = None


class ScopedAsk(BaseModel):
    """The scope a page was ASKED for, echoed back on the answer.

    Echoed rather than inferred because a page can land after the operator
    collapsed the group that asked for it: the client must be able to attribute
    an answer to the request that produced it from the answer itself, not from
    the ordering of its own promises (two scope fetches can be in flight, and the
    one that returns first is not necessarily the one that asked first).

    ``name`` is the display name the request carried -- NOT a live team or
    profile: the operator renames and deletes teams, and a session's stored
    ``attachment.json`` keeps the name it was attached under.
    (``resume.read_session_attachment``'s docstring is explicit that the stored
    name is a historical fact.)
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["team", "agent"]
    name: str


class ScopeTotal(BaseModel):
    """One group's conversation count, for a collapsed row's badge.

    ``active`` counts the group's rows that the listing files under Active
    (``CatalogEntry.active``: pending, unseen, or live), so a group's badge can
    carry both numbers without the client recomputing either.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["team", "agent"]
    name: str
    total: int
    active: int


class ScopeCounts(BaseModel):
    """How many conversations the listing holds, in total and per group.

    Present only when the request asked for it (``with_counts=true``), because
    the census costs one ``attachment.json`` read per visible session -- memoized
    in ``session.catalog._BINDING_MEMO``, so a second call on an unchanged store
    pays only stats, but a cold one is a real cost on a large store and a client
    that does not draw counts must not pay it.

    ``unbound`` is its own number rather than a ``scopes`` entry: a session with
    no ``attachment.json`` binding belongs to no group, and folding it into one
    would invent a group the renderer never draws.

    ``scopes`` is sorted by ``(-total, kind, name)`` so two reads of one store
    cannot order the same counts differently.
    """

    model_config = ConfigDict(extra="forbid")
    total: int
    active: int
    unbound: int
    scopes: list[ScopeTotal]


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

    ``sessions`` IS MORE THAN THE PAGE, deliberately, and the consequence is
    part of the contract: **``len(sessions)`` may exceed ``limit``**. It carries
    the newest ``limit`` conversations AND every pinned conversation the page did
    not reach, because it is the array a client REPLACES its rows with — a pinned
    conversation parked in a sibling field would be one the client does not hold
    until it learns about that field, and on a store larger than the page (the
    ordinary case; 5,267 sessions against a 500-row page on the operator's) the
    pin would then have no row, no count and no trace anywhere in the app. The
    additions are ordinary rows in the catalogue's own ranking order — the page's
    order continued below the page, NOT pin recency, which the store also holds
    and which would put a second ordering authority inside one section — each
    with ``pinned: true``, so a client sections them with no new field and no
    sort. ``limit`` and ``truncated`` describe the PAGE ONLY.

    The CLIENT half of that decision cannot be tested from this repository: there
    is no in-tree consumer of this route, so what is pinned here is the shape the
    app is handed, not the rendering it does with it. The boundary is worth
    knowing before someone reads a green suite as coverage of the feature.
    """

    sessions: list[SessionRow]
    #: Whether the ranking held more rows than the PAGE — the same question it
    #: has always answered, and deliberately not "rows this answer does not
    #: carry": the extras appended for pins are rows the client is being handed,
    #: not history it is missing, and this flag is what a client uses to decide
    #: whether more exists to fetch.
    truncated: bool = False
    limit: int = 100
    degraded: list[str] = Field(default_factory=list)
    #: The position to resume this scope from, or ``None`` when this page is the
    #: last one. OPAQUE to the client: it is the ranking tuple of the page's last
    #: row plus the scope it was minted for, and the client only ever hands it
    #: back.
    #:
    #: INVARIANT, asserted by tests: ``(next_cursor is not None) == truncated``.
    #: ``next_cursor`` is not a restatement of ``truncated`` — ``truncated`` is a
    #: boolean and this is the value needed to fetch the next page — which is why
    #: no separate ``has_more`` was added beside it.
    #:
    #: Minted from the last row of the PAGE and never from an appended off-page
    #: pin, because a pin below the page is not a position in the scope's
    #: continuation: resuming from it would skip every row between the page's end
    #: and that pin.
    next_cursor: str | None = None
    #: Your cursor could not be used, so THIS ANSWER IS THE SCOPE'S FIRST PAGE.
    #:
    #: Not an error: a malformed token, one from an unknown version, and one
    #: minted for a different scope are all answered the same way, and the remedy
    #: is the same for all three — re-read from the top. This mirrors
    #: ``HistoryPage.cursor_missing`` exactly (``server.utils.desktop_sessions``),
    #: including its reasoning that a client which lost its place must be able to
    #: tell "the store moved" from "your token is unusable".
    cursor_missing: bool = False
    #: The scope this page was asked for, or ``None`` for the head listing.
    scope: ScopedAsk | None = None
    #: The per-group census, present only when ``with_counts=true`` asked for it.
    counts: ScopeCounts | None = None

    # THE FOUR FIELDS ABOVE ARE ADDITIVE AND ALWAYS PRESENT, which is the whole
    # compatibility promise stated precisely: a request that sends none of
    # ``scope_kind``/``scope_name``/``cursor``/``with_counts`` gets exactly the
    # ``sessions``/``truncated``/``limit``/``degraded`` it has always got, and the
    # four new fields sit at their defaults (``null``, ``false``, ``null``,
    # ``null``). PRESENT rather than omitted, on the precedent the ``degraded``
    # comment above states: a client must be able to tell "this answer is not
    # paged" from "this server is too old to page it", and the capability map's
    # ``session_catalogue_page`` key is what makes that question askable BEFORE
    # the request rather than only after it.
    #
    # A DAEMON THAT PREDATES THESE FIELDS IS INDISTINGUISHABLE FROM AN ABSENT
    # KEY: every one is defaulted, so a client that never learned the key sees the
    # same listing it always saw.


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
    #: Whether this conversation is pinned: ALWAYS PRESENT, both values, on every
    #: row — the same rule, and the same reason, as :attr:`SessionRow.pinned`.
    #:
    #: NOT OPTIONAL HERE EVEN THOUGH THE SEARCH IS A DIFFERENT QUESTION. A client
    #: that synthesises a row from a search hit — which the app does, for a
    #: conversation beyond the 500 rows its own page holds — would otherwise
    #: render a pinned conversation in an ordinary section with no Pinned section
    #: at all, and offer a pin control whose press is an idempotent no-op that
    #: cannot repair the row (the pin is already true server-side, so the next
    #: search answers the same way). An absent key there is not a neutral choice:
    #: the client reads it as "no claim", so the omission is what makes the state
    #: permanently wrong on that surface.
    #:
    #: REQUIRED rather than defaulted, like the list row's: a projection that
    #: forgot it fails loudly here instead of shipping an omission the client is
    #: entitled to read as no claim.
    pinned: bool
    #: Whether this conversation is archived: ALWAYS PRESENT, both values, on
    #: every hit — `pinned`'s rule, and this key's own reason for existing is
    #: stronger here than on the catalogue.
    #:
    #: A SEARCH HIT IS THE ONE ROW A CLIENT CAN SYNTHESISE THAT THE CATALOGUE
    #: NEVER SENT, so the archived fact has to travel with it: the store's
    #: default search does not return archived conversations at all, which means
    #: every hit a client sees from a default search is ``false`` — and a hit
    #: from an ``include_archived=true`` search is the only way it learns
    #: otherwise. Dropping the key would make the two answers indistinguishable
    #: to a client merging hits into the rows it holds.
    archived: bool


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


#: Why a child job's live window could not be handed over, when its absence
#: needs naming. A TOKEN, not a sentence: the copy belongs to the surface.
#:
#: ``no-owner`` is the runtime half — no owner is attached, the owner is too old
#: for the subscription op, or the connection dropped while the window was being
#: fetched. It is RETRYABLE: the reader re-asks on its next pulse. ``unsupported``
#: is the job half — this job type records no trajectory at all, so there is
#: nothing to follow and the reader must stop asking. The two must not be folded
#: into one: a reader cannot tell them apart from the rows alone (both are empty),
#: and polling forever on ``unsupported`` is the same defect as giving up on
#: ``no-owner``.
ChildTrajectoryUnavailable = Literal["no-owner", "unsupported"]


class ChildTrajectoryWindow(BaseModel):
    """One child job's retained trajectory window, seeded by a subscribe.

    The desktop reader's live path (design § 1, D1.1): ``POST``ing this route
    both loads the window and subscribes the connection to its appends, so the
    reply IS the seed rather than an acknowledgement of one. A client that had
    to fetch and then subscribe would lose whatever landed between the two calls.

    The fields are the runtime's own fetch op's (``job_trajectory``) plus the two
    that only a failed load needs, and the identity rule is why ``base_seq`` is
    here at all: the rows rotate out of the front of a bounded window, so list
    position is not identity — ``base_seq`` is the ``_lo_seq`` stamp of
    ``rows[0]``, and a reader applies an append iff its stamp is greater than the
    watermark this seed established. That is what makes the seed/append
    interleave safe without the reader having to care which arrived first.

    ``total`` and ``trajectory_length`` BOTH count the window this reply carries,
    and they are equal by construction: after a successful load the reply IS the
    whole retained window (the pager loops until the owner's ``total`` is
    reached), so "what this reply carries" and "what the runtime retains for the
    job" are the same window — and the roster row's own ``trajectory_length``,
    which is the follower's COPY of the runtime's number, is not reused for it.
    That copy drifts in both directions and the reply must not: it lags the window
    this call just read, and it is inflated by the duplicated rows the seed
    deduplicates (measured: a reply publishing it reported a count matching
    neither its rows nor the runtime's, in a window that was OVER-counted rather
    than partial). A client that wants the runtime's own figure reads the roster
    row it already has; the reply is about the rows it hands over.

    ``watchers`` and ``joined`` say what this call did to the session's shared
    count, because every POST increments and one DELETE releases one: a client
    that re-seeds without unmounting (a rotation, a double mount) accumulates a
    reference it cannot otherwise observe. ``joined`` is the boolean form of
    ``watchers > 1``. An unavailable reply carries ``0`` and ``False``.
    """

    rows: list[dict[str, Any]] = Field(default_factory=list)
    base_seq: int | None = None
    total: int = 0
    trajectory_length: int = 0
    watchers: int = 0
    joined: bool = False
    available: bool
    reason: ChildTrajectoryUnavailable | None = None


class ChildTrajectoryRelease(BaseModel):
    """What one reader's release left behind for a child job's window.

    ``watchers`` is the count still held on the session's shared bridge, and
    ``watching`` is its nonzero form for a client that only wants the boolean.
    Both are published because the count is the fact and the boolean is the
    convenience: a client that had assumed its release was the last one can see
    that another window is still reading the same child, which is the one case
    where nothing about its own close is observable on screen.
    """

    watching: bool
    watchers: int


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


class PinState(BaseModel):
    """What a session's pin is, after the write that set it.

    Typed rather than ``dict[str, Any]`` for the reason ``AttentionState``
    gives below: the renderer's hand-written copy of this shape cannot drift
    from the authority silently.

    ``pinned`` echoes the state the caller ASKED for, not delta information —
    the request carries a desired state, so an idempotent retry returns exactly
    what the first call returned and the client can reconcile on it.
    """

    session_id: str
    pinned: bool


class ArchiveState(BaseModel):
    """What a session's archive is, after the write that set it.

    ``set_pin``'s shape deliberately, down to the field names: this is the same
    kind of verb on the same kind of address (a per-session flag the client
    reconciles its row on), so a renderer's pin handler and its archive handler
    are the same handler with a different flag rather than two conventions.

    Typed rather than ``dict[str, Any]`` for the reason :class:`PinState` gives:
    the renderer's hand-written copy of this shape cannot drift from the
    authority silently.

    ``archived`` echoes the state the caller ASKED for, not delta information —
    the request carries a desired state, so an idempotent retry returns exactly
    what the first call returned.
    """

    session_id: str
    archived: bool


class DeletedSession(BaseModel):
    """What a completed deletion reports: the id, and that it happened.

    NO CHILD COUNT HERE, deliberately, and the omission is the interface rather
    than an oversight: this answer is receipt-free and idempotent-adjacent (a
    retry after a lost response answers 404, which the client reads as "it is
    gone"), and every additional field is a field a retry cannot reproduce. The
    blast radius belongs in the CONFIRMATION a user reads before the request,
    which is where the TUI states it in words (see ``OperatorApp._cmd_delete``).

    ``deleted`` is always true: this model is only constructed on the success
    path, and a refusal is a 409 carrying a sentence instead.
    """

    session_id: str
    deleted: bool


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
