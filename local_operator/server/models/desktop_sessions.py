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


class CommandReceipt(BaseModel):
    command: str
    result: OwnerCommandResult
    replayed: bool = False


class AnswerReceipt(BaseModel):
    detail: str


class WatchReceipt(BaseModel):
    lease_seconds: Literal[45]
