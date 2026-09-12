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
    sessions: list[SessionRow]
    truncated: bool = False
    limit: int = 100


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


class SnapshotPayload(BaseModel):
    frontend: FrontendSync
    history: HistoryPage
    cold: bool


class SessionSnapshot(BaseModel):
    session_id: str
    epoch: str
    seq: int = Field(ge=0)
    type: Literal["snapshot"]
    payload: SnapshotPayload


class AdmissionDetail(BaseModel):
    status: Literal["admitted"]
    duplicate: bool
    detail: str


class MessageAdmission(AdmissionDetail):
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
