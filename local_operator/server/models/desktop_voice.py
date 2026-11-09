"""Wire models for ``/v1/desktop/voice*`` — the frozen contract of
docs/design/qwen-voice.md ("Frozen contract").

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


class VoiceSettings(BaseModel):
    max_spoken_chars: int


class VoiceStatus(BaseModel):
    """``GET /v1/desktop/voice``."""

    active: VoiceActive | None
    settings: VoiceSettings


class VoiceControl(BaseModel):
    """``POST /v1/desktop/voice`` body."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["enable", "disable"]
    session_id: str = Field(min_length=1, max_length=128)


class VoiceControlResult(BaseModel):
    """``POST /v1/desktop/voice`` 200: the slot after the op, and who it displaced."""

    active: VoiceActive | None
    replaced: str | None = None


class VoiceUtteranceResult(BaseModel):
    """``POST /v1/desktop/voice/utterance`` 200.

    ``path`` is the ``text_origin`` the caller named for the words (``qwen_s2s``:
    the rig's Parakeet final), or ``None`` for the plain text door.
    """

    text: str
    path: str | None
    admission: MessageAdmission


class MessageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["message"]
    message_id: str = Field(min_length=1, max_length=128)


class GateSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["gate"]
    request_id: str = Field(min_length=1, max_length=128)


#: CLOSED, and with no text-bearing member: a source is a REFERENCE the server
#: resolves itself, so ``/voice/text`` cannot be made to return words it was handed.
VoiceTextSource = Annotated[Union[MessageSource, GateSource], Field(discriminator="kind")]


class VoiceTextRequest(BaseModel):
    """``POST /v1/desktop/voice/text`` body: a source reference, never free text (V8)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    source: VoiceTextSource


class VoiceText(BaseModel):
    """``POST /v1/desktop/voice/text`` 200: the words to speak, after ``speakable``.

    ``spoken_chars`` is ``len(text)``, carried so a client can log what was capped
    without recounting.
    """

    text: str
    spoken_chars: int
