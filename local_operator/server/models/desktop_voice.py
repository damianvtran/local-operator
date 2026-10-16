"""Wire models for ``/v1/desktop/voice*`` — the frozen contract of
docs/design/session-voice.md ("Frozen contract (backend ↔ UI)").

The field names and shapes here ARE that contract: the UI repo builds against
them, so a change is a change to the design record first.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from local_operator.server.models.desktop_sessions import MessageAdmission


class VoiceActive(BaseModel):
    """The enabled session: ``{session_id, enabled_at}`` (epoch milliseconds)."""

    session_id: str
    enabled_at: int


class VoicePathState(BaseModel):
    """One direction's resolution, as the status read reports it.

    ``path`` is the resolver's token (``AudioPath`` for STT, ``VoicePath`` for
    TTS), carried as a string so the two closed vocabularies need no shared
    enum. For TTS it is the rung the voice routes will ACTUALLY use, including
    the opt-in ``local_tts_system``.
    """

    path: str
    available: bool
    reason: str


#: The listening front ends (R8). ``elevenlabs`` joins later; the vocabulary is
#: closed so an unknown id is a 422, not a silently ignored field.
FrontEndId = Literal["local", "gemini_live"]


class VoiceFrontEnd(BaseModel):
    """One front end's availability, as the status read reports it (R8)."""

    id: FrontEndId
    available: bool
    reason: str


class VoiceSettings(BaseModel):
    max_spoken_chars: int
    front_end: FrontEndId


class VoiceStatus(BaseModel):
    """``GET /v1/desktop/voice``."""

    active: VoiceActive | None
    stt: VoicePathState
    tts: VoicePathState
    front_ends: list[VoiceFrontEnd]
    settings: VoiceSettings


class VoiceControl(BaseModel):
    """``POST /v1/desktop/voice`` body."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["enable", "disable"]
    session_id: str = Field(min_length=1, max_length=128)
    #: Optional on enable: chooses (and persists) the front end. Absent keeps
    #: the configured one. Ignored on disable.
    front_end: FrontEndId | None = None


class VoiceControlResult(BaseModel):
    """``POST /v1/desktop/voice`` 200: the slot after the op, and who it displaced."""

    active: VoiceActive | None
    replaced: str | None = None


class VoiceUtteranceResult(BaseModel):
    """``POST /v1/desktop/voice/utterance`` 200.

    ``path`` is the STT rung that produced ``text``, the front end's
    ``text_origin`` when it supplied the words, or ``None`` for the plain text
    door. ``control`` is the R6 action a ``controls=1`` take matched; a control
    take has ``admission: null`` because nothing was admitted.
    """

    text: str
    path: str | None
    admission: MessageAdmission | None
    control: str | None = None


class MessageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["message"]
    message_id: str = Field(min_length=1, max_length=128)


class GateSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["gate"]
    request_id: str = Field(min_length=1, max_length=128)


class AskSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["ask"]
    ask_id: str = Field(min_length=1, max_length=128)


class PhraseSource(BaseModel):
    """A fixed catalogue phrase (R5). ``key`` is checked against the closed
    ``wire.voice.phrase.*`` set by the route: an unknown key is 404
    ``unknown_source``, so this source cannot carry free text."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["phrase"]
    key: str = Field(min_length=1, max_length=64)


SpeechSource = Annotated[
    Union[MessageSource, GateSource, AskSource, PhraseSource], Field(discriminator="kind")
]


class VoiceLiveToken(BaseModel):
    """``POST /v1/desktop/voice/live-token`` 200 (R2). Main only; never logged."""

    token: str
    expires_at: str
    new_session_expires_at: str
    model: str


class VoiceSpeechRequest(BaseModel):
    """``POST /v1/desktop/voice/speech`` body: a source REFERENCE, never free text (V8)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    source: SpeechSource
