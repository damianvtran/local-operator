"""Additive HTTP response models over the canonical runtime's own state schema.

Keep FrontendSync and SlashResult shared with attach clients. A parallel HTTP
projection would drop new owner fields and turn unknown accounting into zeros.
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


class CreatedSession(BaseModel):
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
    #: ``[published, acknowledged]`` -- monotonic, and independent of the owner
    #: epoch, which is why a client merges on it rather than on arrival order.
    revision: list[int]
    #: Absent on the cold list path; only a live owner can answer it.
    supported: bool | None = None
