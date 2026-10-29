"""The audio lane's frozen request models: tool params, wire builders, job spec.

Two layers live here, deliberately in one module so they cannot drift:

1. **Tool params** (pydantic) — the exact schema the model sees, frozen by the
   manager's brief (2026-10-17): ``generate_voiceover`` takes text 1..10,000
   chars, an optional ``voice`` (``female|male|<vendor id>``), tone,
   expressiveness, pace 0.5..2.0, language (ISO 639-1), accent, instructions
   (≤8,000 chars) and a model id; ``generate_sound_effect`` takes a prompt
   1..600 chars, a REQUIRED ``duration_s`` 0.5..30, ``loop`` and
   ``prompt_influence`` 0..1. The optional ``sounds`` batch is exclusive with
   the single prompt/duration and shares these generation settings. The class
   docstrings render verbatim into the wire schema's ``description``, so they
   stay one line and the constraints live in field descriptors and here.

2. **Wire builders** — the ONE place the tool params become the server/BYOK
   request bodies. The server contract (frozen v1, Radient Agent Server lane,
   session 6c9c8f09af70) takes the tool params essentially 1:1; the BYOK legs
   map them onto the vendors' own shapes (ElevenLabs ``voice_settings`` /
   ``/v1/sound-generation``; OpenAI ``/v1/audio/speech``). Keeping the
   mapping here means a contract amendment is one edit in one file, and the
   approval describer and the rungs necessarily agree about what will be sent.

**No provider-specific field rides the tool schema or the UI** (frozen rule):
``profile``, tone, expressiveness and friends are the platform's canonical
vocabulary; ``provider``/``model`` are opaque display strings. Vendor-specific
translation (OpenAI's voice names, ElevenLabs' stability/style knobs) happens
ONLY in the BYOK builders/rungs, recorded honestly in the voicing notes.

``AudioJobSpec`` is the kind-neutral walk's input in the audio binding: one
frozen dataclass carrying the tool name, the direction ("voiceover" or
"sound_effect") and every canonical param, so the rungs can be exercised
directly in tests without a pydantic round trip, and the cascade never parses
tool args.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_operator.artifacts import MediaAsset

__all__ = [
    "AudioAsset",
    "AudioJobSpec",
    "GenerateSoundEffectParams",
    "SoundEffectRequest",
    "GenerateVoiceoverParams",
    "SOUND_EFFECT_PROMPT_MAX_CHARS",
    "VOICEOVER_TEXT_MAX_CHARS",
    "default_voiceover_model",
    "sound_effect_request_body",
    "voiceover_request_body",
]

#: The frozen text bound (manager brief / server contract v1): 1..10,000 chars.
VOICEOVER_TEXT_MAX_CHARS = 10_000
#: The frozen instructions bound: ≤8,000 chars (server contract v1).
VOICEOVER_INSTRUCTIONS_MAX_CHARS = 8_000
#: The frozen sound-effect prompt bound: 1..600 chars.
SOUND_EFFECT_PROMPT_MAX_CHARS = 600
#: The frozen sound-effect duration range, seconds. REQUIRED — the server
#: contract has no auto-duration, so the tool has no default for it.
SOUND_EFFECT_DURATION_MIN_S = 0.5
SOUND_EFFECT_DURATION_MAX_S = 30.0

#: The platform's own model ids (server contract v1): ``radient/voiceover`` is
#: multilingual v2 underneath (language is IGNORED — it pronounces whatever
#: it is given) and ``radient/voiceover-flash`` is the flash tier (language is
#: honoured). The ids ride the hub request verbatim; the hub resolves them.
RADIENT_VOICEOVER_MODEL = "radient/voiceover"
RADIENT_VOICEOVER_FLASH_MODEL = "radient/voiceover-flash"

#: The BYOK legs' default models — what an unpinned ``model`` runs, and what
#: the central rate table prices. ElevenLabs' multilingual v2 is the hub's own
#: default for the same delivery property the TTS lane documents; OpenAI's
#: ``gpt-4o-mini-tts`` is the current steerable-and-cheap default.
ELEVENLABS_VOICEOVER_MODEL = "eleven_multilingual_v2"
ELEVENLABS_VOICEOVER_FLASH_MODEL = "eleven_flash_v2_5"
ELEVENLABS_SOUND_EFFECT_MODEL = "eleven_text_to_sound_v2"
OPENAI_VOICEOVER_MODEL = "gpt-4o-mini-tts"

#: ``voice``'s two aliases resolve to vendor ids per BYOK leg. ElevenLabs'
#: ids are its two canonical premade voices (Rachel / Adam) — the ids every
#: ElevenLabs example uses; OpenAI's are the published voice names.
ELEVENLABS_VOICE_ALIASES = {"female": "21m00Tcm4TlvDq8ikWAM", "male": "pNInz6obpgDQGcFmaJgB"}
OPENAI_VOICE_ALIASES = {"female": "nova", "male": "onyx"}


class GenerateVoiceoverParams(BaseModel):
    """Arguments for the ``generate_voiceover`` tool."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=VOICEOVER_TEXT_MAX_CHARS,
        description="The words to speak (1-10,000 characters).",
    )
    voice: str | None = Field(
        default=None,
        description="female|male, or a provider voice id; the default voice when omitted.",
    )
    tone: Literal["warm", "neutral", "bright", "calm", "authoritative"] | None = Field(
        default=None, description="Delivery tone."
    )
    expressiveness: Literal["low", "medium", "high"] | None = Field(
        default=None, description="How animated the delivery is."
    )
    pace: float | None = Field(
        default=None, ge=0.5, le=2.0, description="Speaking rate 0.5-2.0 (1.0 = normal)."
    )
    language: str | None = Field(
        default=None, description="ISO 639-1 language tag, where the model honours it."
    )
    accent: str | None = Field(default=None, description="Accent hint, where supported.")
    instructions: str | None = Field(
        default=None,
        max_length=VOICEOVER_INSTRUCTIONS_MAX_CHARS,
        description="Free-form delivery direction (up to 8,000 characters).",
    )
    model: str | None = Field(
        default=None, description="Provider model id; its default when omitted."
    )
    filename: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Filename stem: a short lowercase hyphenated description of the request, "
            "no extension (e.g. `welcome-message`); omit to derive it from the prompt."
        ),
    )


class SoundEffectRequest(BaseModel):
    """One named batch request; provider rejection remains an item-level result."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "This item's filename stem, lowercase hyphenated, no extension "
            "(e.g. `bird-chirping-sound-effect`); defaults to the prompt."
        ),
    )
    prompt: str = Field(min_length=1, max_length=SOUND_EFFECT_PROMPT_MAX_CHARS)
    # Batch duration bounds belong to the provider: rejecting a single item's
    # duration here would discard every valid sibling before the paid call.
    duration_s: float = Field(allow_inf_nan=False)


class GenerateSoundEffectParams(BaseModel):
    """Use prompt + duration_s for one sound, or sounds for a batch (1-50); never both."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=SOUND_EFFECT_PROMPT_MAX_CHARS,
        description="Single form: required with duration_s; omit with sounds (1-600 characters).",
    )
    duration_s: float | None = Field(
        default=None,
        ge=SOUND_EFFECT_DURATION_MIN_S,
        le=SOUND_EFFECT_DURATION_MAX_S,
        description="Single form: required with prompt (0.5-30 s); omit with sounds.",
    )
    loop: bool = Field(default=False, description="Make the clip loop seamlessly.")
    prompt_influence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="How literally the prompt is followed (0-1)."
    )
    model: str | None = Field(
        default=None, description="Provider model id; its default when omitted."
    )
    filename: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "Single form's filename stem, lowercase hyphenated, no extension "
            "(e.g. `bird-chirping-sound-effect`); omit to derive it from the prompt."
        ),
    )

    sounds: list[SoundEffectRequest] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description=(
            "Named sound effects in one approval (1-50); exclusive with "
            "prompt/duration_s/filename."
        ),
    )

    @model_validator(mode="after")
    def exactly_one_form(self) -> GenerateSoundEffectParams:
        if self.sounds is not None:
            if {"prompt", "duration_s", "filename"} & self.model_fields_set:
                raise ValueError(
                    "Use either sounds or prompt + duration_s (+ optional filename), not both"
                )
        elif self.prompt is None or self.duration_s is None:
            raise ValueError("Supply either sounds or both prompt and duration_s")
        return self


@dataclass(frozen=True)
class AudioJobSpec:
    """One audio request in the walk's kind-neutral form.

    ``tool`` is the emitting tool's name (``generate_voiceover`` /
    ``generate_sound_effect``) — pinned here so every progress update and
    attempt record names the tool the caller actually invoked. ``direction``
    is the lane's own split ("voiceover" / "sound_effect"); the rung filter
    reads it, the walk never does.
    """

    tool: str
    direction: Literal["voiceover", "sound_effect"]
    #: The spoken text / the sound's prompt — also the walk's ``JobSpec.prompt``
    #: echo, so a receipt can repeat the request without the model remembering.
    text: str
    voice: str | None = None
    tone: str | None = None
    expressiveness: str | None = None
    pace: float | None = None
    language: str | None = None
    accent: str | None = None
    instructions: str | None = None
    model: str | None = None
    #: Sound-effect fields (unused for a voiceover).
    duration_s: float | None = None
    # Only batched submits jitter admission retries; preserve single-call timing.
    batch_submission: bool = False
    loop: bool = False
    prompt_influence: float | None = None


@dataclass(frozen=True)
class AudioAsset(MediaAsset):
    """One audio asset: the generic asset plus the facts an audio surface needs.

    Subclasses :class:`~local_operator.artifacts.MediaAsset` so it travels the
    generic walk's ``assets`` list unchanged, and the tool reads the extra
    fields with ``getattr`` (absent-safe: a BYOK leg knows fewer facts — the
    rule is "absent where unknown", never a synthesized stand-in). ``sha256``
    is what a Radient download was VERIFIED against; ``None`` where no
    provider digest existed to verify. THE ONE FACT THAT IS NOT ABSENT-SAFE is
    ``duration_s``: a surface needs a length to draw a waveform and a clock,
    and a record without one renders NO PLAYER, so it is derived from the
    bytes where a vendor states none (``audiogen.peaks.compute_duration``).
    """

    #: Container ("mp3"), when the provider stated one.
    format: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    bitrate_kbps: int | None = None
    #: The provider's own model id for what ran (Radient's ``provider_model_id``).
    provider_model_id: str | None = None
    #: The signed URL's expiry (ISO 8601 / epoch string, verbatim from the wire).
    expires_at: str | None = None
    sha256: str | None = None
    #: ``voicing.notes`` (Radient) / the BYOK mapping's own record:
    #: ``{applied: [...], ignored: [...], clamped: [...]}``.
    voicing_notes: dict[str, Any] | None = field(default=None)


def default_voiceover_model(route: str) -> str:
    """The model an unpinned voiceover runs on ``route``.

    ONE resolution read by the rungs (what actually runs), the pricing table
    (what the estimate prices) and the approval describer (what the person
    approves) — three consumers of one answer, so they cannot disagree. A
    route this map does not know answers with the Radient id, which is the
    cascade's first rung and the only unpinned case the server contract
    defines.
    """
    if route == "elevenlabs":
        return ELEVENLABS_VOICEOVER_MODEL
    if route == "openai":
        return OPENAI_VOICEOVER_MODEL
    return RADIENT_VOICEOVER_MODEL


def voiceover_request_body(spec: AudioJobSpec) -> dict[str, Any]:
    """The Radient v1 voiceover request body, mapped from the canonical params.

    The frozen contract takes the canonical fields essentially 1:1 — ``voice``
    travels verbatim (``female``/``male``/a vendor id; the server maps it to
    its canonical v1 voice descriptor and reports what it applied, ignored or
    clamped in ``voicing.notes``). Optional fields ride only when set: an
    omitted field is "no preference", never a default the caller did not
    choose. No API key ever rides this body (frozen rule: the server route
    never accepts a provider key).
    """
    body: dict[str, Any] = {
        "model": spec.model or RADIENT_VOICEOVER_MODEL,
        "text": spec.text,
    }
    for key in ("voice", "tone", "expressiveness", "pace", "language", "accent", "instructions"):
        value = getattr(spec, key)
        if value is not None:
            body[key] = value
    return body


def sound_effect_request_body(spec: AudioJobSpec) -> dict[str, Any]:
    """The Radient v1 sound-effect request body.

    ``duration_seconds`` is REQUIRED by the contract (no auto-duration), and
    the tool has no default for it either — this builder asserts the spec
    carries one rather than inventing a length nobody chose. ``model`` rides
    only when the caller PINNED one: unlike voiceover, the contract names no
    default sound-effect model id, so the unpinned call lets the server
    resolve its own (the response reports which ran) instead of this client
    inventing a spelling for it.
    """
    if spec.duration_s is None:  # pragma: no cover - the tool validates first
        raise ValueError("sound effects require a duration_s (no auto-duration).")
    body: dict[str, Any] = {
        "prompt": spec.text,
        "duration_seconds": float(spec.duration_s),
        "loop": bool(spec.loop),
    }
    if spec.model:
        body["model"] = spec.model
    if spec.prompt_influence is not None:
        body["prompt_influence"] = float(spec.prompt_influence)
    return body
