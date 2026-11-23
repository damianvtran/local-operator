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


class VoiceByok(BaseModel):
    """Whether bring-your-own-key voice can run, for the one provider supported.

    ``ready`` is the cascade's own answer for a request PINNED to ``provider``
    (``only=``), so it is false when only another rung has a credential (a Radient
    login, a Groq key stored outside the desktop picker): pinned voice never falls
    through to those. ``reason`` is the resolver's sentence for the pinned rung,
    ready or not.
    """

    provider: Literal["elevenlabs"]
    ready: bool
    reason: str


class VoiceProviderState(BaseModel):
    """Whether one provider can serve voice RIGHT NOW, from stored credentials only.

    ``ready`` needs BOTH halves (hearing and speaking) on that provider. ``reason``
    is the resolver's own sentence for the half that is missing, and ``null`` when
    ready. Computed without a network call or any spend.
    """

    ready: bool
    reason: str | None


class VoiceProviders(BaseModel):
    radient: VoiceProviderState
    elevenlabs: VoiceProviderState


class VoiceStatus(BaseModel):
    """``GET /v1/desktop/voice``.

    ``provider`` is the user's ``speech.provider`` choice (``null`` = none chosen
    yet, which makes the voice routes refuse); ``providers`` says whether each
    choice could serve. ``byok`` is kept for the app that predates the choice and
    is additive and optional: a client that does not know it ignores it.
    """

    active: VoiceActive | None
    settings: VoiceSettings
    byok: VoiceByok | None = None
    provider: Literal["radient", "elevenlabs"] | None = None
    providers: VoiceProviders | None = None


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
    the rig's Parakeet final), or ``None`` for the plain text door. For a
    recording it is the STT rung's token (``provider_stt_groq``, ...), recorded
    as the message's ``input_path``.
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


class PhraseSource(BaseModel):
    """A fixed catalogue phrase. ``key`` is checked against the closed
    ``wire.voice.phrase.*`` set by the route: an unknown key is 404
    ``unknown_source``, so this source cannot carry free text."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["phrase"]
    key: str = Field(min_length=1, max_length=64)


#: ``/voice/speech`` adds the phrase source; ``/voice/text`` does not (a phrase
#: is a fixed sentence the client already knows by key, so there is nothing to
#: resolve or segment).
SpeechSource = Annotated[
    Union[MessageSource, GateSource, PhraseSource], Field(discriminator="kind")
]


class VoiceTextRequest(BaseModel):
    """``POST /v1/desktop/voice/text`` body: a source reference, never free text (V8)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    source: VoiceTextSource


class VoiceText(BaseModel):
    """``POST /v1/desktop/voice/text`` 200: the words to speak, after ``speakable``.

    ``spoken_chars`` is ``len(text)``, carried so a client can log what was capped
    without recounting. ``truncated`` says whether the cap actually cut this
    reply, so a client can show the cut (a speaker cannot see it) without
    comparing lengths against a cap the backend may have clamped.

    ``full_text`` is the reply cleaned for DISPLAY and never capped: the same
    markup removal as ``text`` (code blocks are the placeholder) but with its
    lines kept, table rows joined by `` · `` and web addresses as written, so a
    screen can show the whole reply while the audio stays capped. It excludes the
    spoken closing sentence. Additive: a client must tolerate an older server
    omitting it.
    """

    text: str
    spoken_chars: int
    truncated: bool
    segments: list[str]
    full_text: str


class VoiceSpeechRequest(BaseModel):
    """``POST /v1/desktop/voice/speech`` body: a source REFERENCE and a segment index.

    The server re-resolves the source and re-segments it (D3), so no free text
    crosses (V8). ``segment`` indexes the ``segments`` list ``/voice/text``
    returned for the same source; a phrase has exactly one segment, index 0.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    source: SpeechSource
    segment: int = Field(default=0, ge=0, le=1000)
    #: LEGACY, accepted so an older app keeps working, and IGNORED: the provider
    #: is the ``speech.provider`` setting, which wins over whatever is sent here.
    provider: Literal["radient", "elevenlabs"] | None = None
