"""The Python executor of the canonical voicing map.

The map is DATA (``voicing_map/speech_voicing_map.v1.json``), vendored
byte-for-byte from ``radient-ml/agent-server``; this module is a thin executor
of it, and it is pinned against the same golden vectors
(``voicing_map/vectors.v1.json``) that the hub's Go executor runs. Two
implementations of "what does ``tone=calm`` mean" would drift silently and the
drift would show up as audio, which no unit test reads — the shared vectors are
what turn that into a red test instead.

THE TWO RULES THAT KEEP IT HONEST (identical to the hub's):

1. Nothing is dropped silently. Every descriptor field ends up either
   ``applied`` (honoured exactly) or in a :class:`Note` saying what happened
   instead — emulated, clamped, nearest, ignored, or pinned by an explicit
   legacy field. The invariant test asserts exactly that, per provider.
2. Identical ``(descriptor, map_version)`` always yields identical params. A
   change to the output of an existing vector is a MAJOR version bump of the
   map, tested by the vendored vectors.

The daemon maps for its BYO rungs, where its own key is the only credential
that can be used; the hub maps for its own cascade. Both map onto the same
file, which is why the vendored copy exists (see ``voicing_map/README.md`` for
the sync discipline).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

__all__ = [
    "Action",
    "Attempt",
    "Legacy",
    "MapDoc",
    "Note",
    "Params",
    "VENDORED_FROM",
    "VENDORED_VERSION",
    "all_fields",
    "degraded_tokens",
    "format_degraded",
    "is_provider_voice",
    "is_voice_alias",
    "load_map",
    "select_voice_row",
    "resolve",
]

#: The map version this module was vendored against. The conformance test
#: asserts it equals the file's own ``map_version``, so a half-vendored pair
#: (new JSON, stale constant) is a red test rather than a silent skew.
VENDORED_VERSION = "1.0"

#: The ``agent-server`` commit the vendored pair was copied from, kept as a
#: reviewable record of provenance (see ``voicing_map/README.md``). Updated by
#: the same procedure that re-copies the files.
VENDORED_FROM = "agent-server c817d5b (feat/speech-voicing-s1-hub-core, #84)"

#: Where the vendored artifacts live, beside this module so a wheel carries
#: them (``pyproject.toml`` ``[tool.setuptools.package-data]``).
VOICING_MAP_DIR = Path(__file__).resolve().parent / "voicing_map"

#: The descriptor's field list, used by the invariant test (every field applied
#: or noted) and by the daemon's conformance runner.
_FIELD_ORDER: tuple[str, ...] = (
    "gender",
    "tone",
    "expressiveness",
    "pace",
    "language",
    "accent",
    "instructions",
)

#: The actions a :class:`Note` can carry. Mirrors the hub's ``Action``.
Action = Literal["applied", "emulated", "clamped", "nearest", "ignored", "pinned"]

#: The descriptor's accepted shape checks, compiled once. ``resolve`` is on the
#: request path of every descriptor-bearing call.
_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True)
class Note:
    """What actually happened to one descriptor field.

    ``detail`` is a single token (no spaces or commas) because it is also the
    ``=detail`` part of the ``X-Radient-Speech-Degraded`` header.
    """

    field: str
    action: Action
    detail: str = ""

    def to_wire(self) -> dict[str, str]:
        if self.detail:
            return {"field": self.field, "action": self.action, "detail": self.detail}
        return {"field": self.field, "action": self.action}

    def token(self) -> str:
        """The header grammar spelling: ``field:action[=detail]``."""
        return f"{self.field}:{self.action}" + (f"={self.detail}" if self.detail else "")


@dataclass
class Params:
    """The provider-native request fields the mapping produces.

    The union of what either provider uses; the caller sends only the fields
    its provider has. ``None`` means "send nothing at all", which is what a
    pace of exactly 1.0 resolves to so an absent descriptor changes nothing on
    the wire.
    """

    #: ElevenLabs voice id, or OpenAI voice name.
    voice_id: str = ""
    model_id: str = ""
    language_code: Optional[str] = None
    #: OpenAI-only: the caller's text with the emulation sentences composed in.
    instructions: Optional[str] = None
    #: ElevenLabs-only (``voice_settings.stability``), from expressiveness.
    stability: Optional[float] = None
    speed: Optional[float] = None

    def to_wire(self) -> dict[str, Any]:
        """The JSON the golden vectors compare, with the Go omitempty rules.

        Omitted rather than null, because the vectors are the contract between
        the two implementations and Go's ``omitempty`` makes "absent" a
        distinct state from "present and empty".
        """
        payload: dict[str, Any] = {}
        if self.voice_id:
            payload["voice_id"] = self.voice_id
        if self.model_id:
            payload["model_id"] = self.model_id
        if self.language_code:
            payload["language_code"] = self.language_code
        if self.instructions:
            payload["instructions"] = self.instructions
        if self.stability is not None:
            payload["stability"] = self.stability
        if self.speed is not None:
            payload["speed"] = self.speed
        return payload


@dataclass
class Attempt:
    """One provider's mapping of a descriptor, computed at attempt time."""

    provider: str
    map_version: str
    voice_key: str = ""
    params: Params = field(default_factory=Params)
    notes: tuple[Note, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"provider": self.provider}
        if self.voice_key:
            payload["voice_key"] = self.voice_key
        payload["map_version"] = self.map_version
        payload["params"] = self.params.to_wire()
        payload["notes"] = [note.to_wire() for note in self.notes]
        return payload

    def applied_fields(self) -> list[str]:
        """The descriptor fields honoured exactly, for ``X-Radient-Speech-Applied``."""
        return [note.field for note in self.notes if note.action == "applied"]

    def degraded_tokens(self) -> list[str]:
        """Everything not honoured exactly, as ``field:action[=detail]`` tokens.

        Emulated output belongs here rather than in Applied: the caller asked
        for a tone and got a sentence asking the vendor to sound that way,
        which is a real difference in what was sent.
        """
        return [note.token() for note in self.notes if note.action != "applied"]


@dataclass(frozen=True)
class Legacy:
    """The pre-descriptor request fields, which win per field when set.

    A non-zero one is reported as ``pinned``, because a client that spelled out
    a voice id, an instruction string, a speed or a language code meant that
    exact value (see the hub's ``docs/SPEECH.md``, "Voicing").
    """

    voice: str = ""
    instructions: str = ""
    speed: Optional[float] = None
    language_code: str = ""


@dataclass(frozen=True)
class MapDoc:
    """The parsed canonical map, validated once at load."""

    map_version: str
    descriptor_version: int
    capabilities: dict[str, dict[str, dict[str, Any]]]
    voices: tuple[dict[str, Any], ...]
    tone_phrase: dict[str, str]
    expressiveness: dict[str, Any]
    language_phrase: dict[str, str]
    accent_phrase: dict[str, str]
    fixed: dict[str, Any]
    nearest: dict[str, Any]
    #: The compiled ElevenLabs id shape, derived from ``fixed`` at parse time.
    #: A FIELD rather than a module-level side table keyed by ``id(doc)``: an id
    #: is reusable after the object is collected, and a table keyed by one is a
    #: lookup that can answer about a document that no longer exists.
    voice_id_pattern: re.Pattern[str] = None  # type: ignore[assignment]


#: The loaded map, validated at first use. ``load_map`` caches it: the file is
#: a compile-time artifact and reading it per request would put a JSON parse on
#: every spoken message.
_THE_MAP: Optional[MapDoc] = None


def _parse_map(raw: dict[str, Any]) -> MapDoc:
    """Validate and reshape the artifact. Raises ``ValueError`` on a defect.

    A failure here is a defect in a vendored file, not a runtime condition a
    request can cause, so it is loud rather than a degraded response — the same
    reasoning the hub uses for its embedded map.
    """
    version = str(raw.get("map_version", "")).strip()
    if not version:
        raise ValueError("voicing map: map_version is empty")
    voices_raw = raw.get("voices")
    if not isinstance(voices_raw, list) or not voices_raw:
        raise ValueError("voicing map: the voice pool is empty")
    for row in voices_raw:
        missing = [k for k in ("key", "gender", "tones") if not row.get(k)]
        if missing:
            raise ValueError(f"voicing map: voice row {row.get('key')!r} is missing {missing}")
        if not (row.get("elevenlabs") or {}).get("voice_id"):
            raise ValueError(f"voicing map: voice row {row['key']!r} has no elevenlabs id")
        if not (row.get("openai") or {}).get("voice"):
            raise ValueError(f"voicing map: voice row {row['key']!r} has no openai voice")
    fixed = raw.get("fixed") or {}
    pattern = (fixed.get("elevenlabs") or {}).get("voice_id_pattern")
    if not pattern:
        raise ValueError("voicing map: fixed.elevenlabs.voice_id_pattern is missing")
    doc = MapDoc(
        map_version=version,
        descriptor_version=int(raw.get("descriptor_version") or 0),
        capabilities=raw.get("capabilities") or {},
        voices=tuple(voices_raw),
        tone_phrase=raw.get("tone_phrase") or {},
        expressiveness=raw.get("expressiveness") or {},
        language_phrase=raw.get("language_phrase") or {},
        accent_phrase=raw.get("accent_phrase") or {},
        fixed=fixed,
        nearest=raw.get("nearest") or {},
        voice_id_pattern=re.compile(pattern),
    )
    return doc


def load_map(path: Path | None = None) -> MapDoc:
    """The vendored map, parsed and validated once per process.

    ``path`` is a test seam (the conformance runner reads the vendored copy
    through it); production callers take the cache.
    """
    global _THE_MAP
    if path is not None:
        return _parse_map(json.loads(path.read_text(encoding="utf-8")))
    if _THE_MAP is None:
        _THE_MAP = _parse_map(
            json.loads((VOICING_MAP_DIR / "speech_voicing_map.v1.json").read_text(encoding="utf-8"))
        )
    return _THE_MAP


def all_fields() -> list[str]:
    """The descriptor's field list, for the invariant test."""
    return list(_FIELD_ORDER)


def select_voice_row(doc: MapDoc, gender: str, tone: str) -> tuple[dict[str, Any], str]:
    """The documented nearest rule (map ``nearest.voice_row``).

    Exact ``(gender, tone)`` row, else the same gender's ``warm`` row, else the
    same gender's first row, else the female warm row. Returns the row and the
    tone actually used, so the caller can emit the ``nearest`` Note.
    """
    for row in doc.voices:
        if row["gender"] == gender and tone in row["tones"]:
            return row, tone
    for row in doc.voices:
        if row["gender"] == gender and "warm" in row["tones"]:
            return row, "warm"
    for row in doc.voices:
        if row["gender"] == gender:
            return row, row["tones"][0]
    fallback_gender = (doc.nearest.get("gender") or {}).get("unknown", "female")
    for row in doc.voices:
        if row["gender"] == fallback_gender and "warm" in row["tones"]:
            return row, "warm"
    return doc.voices[0], doc.voices[0]["tones"][0]


def is_voice_alias(voice: str) -> bool:
    return voice.strip().lower() in ("female", "male")


def is_provider_voice(provider: str, voice: str) -> bool:
    """Whether a voice identifier is one ``provider`` can actually use.

    One field, two id spaces: a request may pin an ElevenLabs id while being
    served by (or falling back to) OpenAI, and copying the id across would ask
    that vendor for a voice it does not have — on a fallback leg that is the
    loss of the fallback itself.
    """
    if provider == "elevenlabs":
        return bool(load_map().voice_id_pattern.match(voice))
    if provider == "openai":
        names = (load_map().fixed.get("openai") or {}).get("voices") or ()
        return voice.strip().lower() in {name.lower() for name in names}
    return False


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _format_float(value: float) -> str:
    """The short token the header grammar wants: no exponent, no trailing zeros."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _ordered_notes(notes: dict[str, Note]) -> tuple[Note, ...]:
    """Notes in the fixed field order, so a golden vector is a stable comparison."""
    out: list[Note] = []
    seen: set[str] = set()
    for field_name in _FIELD_ORDER:
        note = notes.get(field_name)
        if note is not None:
            out.append(note)
            seen.add(field_name)
    for field_name in sorted(set(notes) - seen):
        out.append(notes[field_name])
    return tuple(out)


def _compose_instructions(base: str, candidates: Sequence[tuple[str, str]]) -> tuple[str, set[str]]:
    """Join the caller's text with our emulation sentences, bounded.

    The cap has to hold on what the VENDOR receives, not only on the field the
    caller filled in: the caller's text is validated at <= 8000 and the
    emulation sentences are OURS on top of it. The caller's text is never
    truncated — our sentences are dropped instead, from the last joined
    backwards — and each dropped axis says so in the degraded channel.
    """
    from local_operator.tts.descriptor import MAX_INSTRUCTIONS

    kept = list(candidates)

    def build() -> str:
        parts = [base, *(text for _field, text in kept)]
        return " ".join(part.strip() for part in parts if part.strip())

    composed = build()
    dropped: set[str] = set()
    index = len(kept) - 1
    while index >= 0 and len(composed) > MAX_INSTRUCTIONS:
        field_name, text = kept[index]
        if not text.strip():
            index -= 1
            continue
        dropped.add(field_name)
        kept.pop(index)
        composed = build()
        index -= 1
    return composed, dropped


def _apply_legacy_pins(
    attempt: Attempt, notes: dict[str, Note], legacy: Legacy, provider: str
) -> None:
    """Let an explicit pre-descriptor field win its own field.

    Existing clients spell out a voice id, an instructions string, a speed or a
    language code; those are exact values, so they must not be reinterpreted
    through the descriptor's semantics. The pinned Note keeps the override
    visible instead of silent.
    """

    def pin(field_name: str, detail: str) -> None:
        notes[field_name] = Note(field_name, "pinned", detail)

    if legacy.voice and not is_voice_alias(legacy.voice):
        if is_provider_voice(provider, legacy.voice):
            # A non-alias voice id names gender and tone itself, so both fields
            # are pinned by it.
            attempt.voice_key = ""
            attempt.params.voice_id = legacy.voice
            pin("gender", "voice")
            pin("tone", "voice")
        else:
            # The pinned voice belongs to ANOTHER provider's id space.
            # Attributed to the legacy `voice` field rather than to
            # gender/tone: those descriptor fields WERE honoured, so a pinned
            # note on them would claim the opposite.
            notes["voice"] = Note("voice", "ignored", provider)
    if legacy.instructions:
        if provider == "openai":
            # The caller's text is used verbatim, so the emulation sentences
            # are not composed in and the axes that would have ridden on them
            # are pinned too.
            attempt.params.instructions = legacy.instructions
            for field_name in ("tone", "expressiveness", "language", "accent"):
                if notes[field_name].action == "emulated":
                    pin(field_name, "instructions")
        pin("instructions", "legacy")
    if legacy.speed is not None:
        # The caller's own speed wins and travels through exactly as today: the
        # per-provider clamp is the provider service's job, not the
        # descriptor's, so the pinned value is not re-clamped here.
        attempt.params.speed = legacy.speed
        pin("pace", "speed")
    if legacy.language_code:
        # language_code is an ElevenLabs field; on the OpenAI leg the caller's
        # value has nowhere to go, so the pinned note is what records that the
        # descriptor's language did not take effect.
        if provider == "elevenlabs":
            attempt.params.language_code = legacy.language_code
        pin("language", "language_code")


def _resolve_elevenlabs(
    doc: MapDoc, provider: str, model: str, descriptor: Any, legacy: Legacy
) -> Attempt:
    notes: dict[str, Note] = {}

    def add(field_name: str, action: Action, detail: str = "") -> None:
        notes[field_name] = Note(field_name, action, detail)

    gender = str(descriptor.gender or "").strip().lower()
    if gender not in ("female", "male"):
        gender = (doc.nearest.get("gender") or {}).get("unknown", "female")
        add("gender", "nearest", gender)
    else:
        add("gender", "applied")

    tone = str(descriptor.tone or "").strip().lower() or "warm"
    if tone not in doc.tone_phrase:
        # A newer client's tone. Degrade, never refuse: the caller asked for
        # audio, not a schema error.
        add("tone", "nearest", (doc.nearest.get("tone") or {}).get("unknown", "neutral"))
        tone = (doc.nearest.get("tone") or {}).get("unknown", "neutral")
    else:
        add("tone", "applied")

    # An alias voice names a gender, so it wins THAT field while the
    # descriptor's tone still drives the row.
    if is_voice_alias(legacy.voice):
        gender = legacy.voice.strip().lower()
        notes["gender"] = Note("gender", "pinned", "voice")

    row, used_tone = select_voice_row(doc, gender, tone)
    if used_tone != tone:
        notes["tone"] = Note("tone", "nearest", used_tone)

    attempt = Attempt(
        provider=provider,
        map_version=doc.map_version,
        voice_key=row["key"],
        params=Params(voice_id=row["elevenlabs"]["voice_id"], model_id=model),
    )

    # expressiveness -> voice_settings.stability.
    level = str(descriptor.expressiveness or "") or "medium"
    stability_map = (doc.expressiveness.get("elevenlabs") or {}).get("stability") or {}
    if level not in stability_map:
        attempt.params.stability = stability_map.get("medium", 0.5)
        add("expressiveness", "nearest", "medium")
    else:
        attempt.params.stability = stability_map[level]
        add("expressiveness", "applied")

    # pace -> voice_settings.speed, clamped to the vendor's range. 1.0 sends
    # nothing at all, so an absent descriptor is byte-identical to today's.
    pace = 1.0 if descriptor.pace is None else float(descriptor.pace)
    el_fixed = doc.fixed.get("elevenlabs") or {}
    if pace != 1.0:
        clamped = _clamp(
            pace, float(el_fixed.get("speed_min", 0.7)), float(el_fixed.get("speed_max", 1.2))
        )
        attempt.params.speed = clamped
        if clamped != pace:
            add("pace", "clamped", _format_float(clamped))
        else:
            add("pace", "applied")
    else:
        add("pace", "applied")

    # language -> language_code, honoured only by the models that take it.
    language = str(descriptor.language or "")
    models = ((doc.capabilities.get("elevenlabs") or {}).get("language") or {}).get("models") or []
    if language in ("", "auto"):
        add("language", "applied", "auto")
    elif model in models:
        attempt.params.language_code = language
        add("language", "applied")
    else:
        add("language", "ignored", model)

    # accent and instructions have no field on this provider.
    if descriptor.accent:
        add("accent", "ignored", "no-field")
    else:
        add("accent", "applied")
    if descriptor.instructions is None or descriptor.instructions == "":
        add("instructions", "applied")
    else:
        add("instructions", "ignored", "no-field")

    _apply_legacy_pins(attempt, notes, legacy, provider)
    attempt.notes = _ordered_notes(notes)
    return attempt


def _resolve_openai(
    doc: MapDoc, provider: str, model: str, descriptor: Any, legacy: Legacy
) -> Attempt:
    notes: dict[str, Note] = {}

    def add(field_name: str, action: Action, detail: str = "") -> None:
        notes[field_name] = Note(field_name, action, detail)

    gender = str(descriptor.gender or "").strip().lower()
    if gender not in ("female", "male"):
        gender = (doc.nearest.get("gender") or {}).get("unknown", "female")
        add("gender", "nearest", gender)
    else:
        add("gender", "applied")

    tone = str(descriptor.tone or "").strip().lower() or "warm"
    if tone not in doc.tone_phrase:
        add("tone", "nearest", (doc.nearest.get("tone") or {}).get("unknown", "neutral"))
        tone = (doc.nearest.get("tone") or {}).get("unknown", "neutral")
    else:
        add("tone", "applied")

    if is_voice_alias(legacy.voice):
        gender = legacy.voice.strip().lower()
        notes["gender"] = Note("gender", "pinned", "voice")

    row, used_tone = select_voice_row(doc, gender, tone)
    if used_tone != tone:
        notes["tone"] = Note("tone", "nearest", used_tone)

    attempt = Attempt(
        provider=provider,
        map_version=doc.map_version,
        voice_key=row["key"],
        params=Params(voice_id=row["openai"]["voice"], model_id=model),
    )

    # pace -> speed, clamped to the vendor's own wider range.
    pace = 1.0 if descriptor.pace is None else float(descriptor.pace)
    oa_fixed = doc.fixed.get("openai") or {}
    if pace != 1.0:
        clamped = _clamp(
            pace, float(oa_fixed.get("speed_min", 0.25)), float(oa_fixed.get("speed_max", 4.0))
        )
        attempt.params.speed = clamped
        if clamped != pace:
            add("pace", "clamped", _format_float(clamped))
        else:
            add("pace", "applied")
    else:
        add("pace", "applied")

    # tone, expressiveness, language and accent are emulated in the
    # instructions text: the vendor takes no parameter for any of them. A field
    # at its neutral default has nothing to emulate, so it reads `applied`.
    level = str(descriptor.expressiveness or "") or "medium"
    phrases = (doc.expressiveness.get("openai") or {}).get("phrase") or {}
    if level not in phrases:
        expression_phrase = phrases.get("medium", "")
        add("expressiveness", "nearest", "medium")
    else:
        expression_phrase = phrases[level]
        if expression_phrase == "":
            add("expressiveness", "applied")
        else:
            add("expressiveness", "emulated", "instructions")

    if tone in doc.tone_phrase:
        # Only an emulation when there is a sentence to append.
        if doc.tone_phrase[tone] == "":
            notes["tone"] = Note("tone", "applied")
        else:
            notes["tone"] = Note("tone", "emulated", "instructions")

    language = str(descriptor.language or "")
    if language in ("", "auto"):
        add("language", "applied", "auto")
        language_phrase = ""
    else:
        language_phrase = doc.language_phrase.get(provider, "")
        add("language", "emulated", "instructions")

    if not descriptor.accent:
        add("accent", "applied")
        accent_phrase = ""
    else:
        accent_phrase = doc.accent_phrase.get(provider, "") % descriptor.accent
        add("accent", "emulated", "instructions")

    base = "" if descriptor.instructions is None else descriptor.instructions
    composed, dropped = _compose_instructions(
        base,
        (
            ("tone", doc.tone_phrase.get(tone, "")),
            ("expressiveness", expression_phrase),
            ("language", language_phrase),
            ("accent", accent_phrase),
        ),
    )
    attempt.params.instructions = composed
    for field_name in dropped:
        notes[field_name] = Note(field_name, "ignored", "instructions-cap")
    add("instructions", "applied")

    _apply_legacy_pins(attempt, notes, legacy, provider)
    attempt.notes = _ordered_notes(notes)
    return attempt


def resolve(provider: str, model: str, descriptor: Any, legacy: Legacy | None = None) -> Attempt:
    """Map a descriptor onto one provider's native params, at attempt time.

    ``provider`` is a key in the map (``elevenlabs``, ``openai``) or
    ``radient``, which is the daemon's own rung: it maps onto the leg the hub
    would try first, because the hub owns leg choice (the map's radient
    capability row states the weakest guarantee across the cascade).

    ``descriptor`` is a :class:`~local_operator.tts.descriptor.VoiceDescriptor`.
    ``None`` is the pre-descriptor behaviour: nothing to map, nothing to report.
    """
    doc = load_map()
    legacy = legacy or Legacy()
    if descriptor is None:
        return Attempt(provider=provider, map_version=doc.map_version)
    if provider in ("elevenlabs", "radient"):
        # `radient` is the hub's own name for its cascade; the primary leg is
        # ElevenLabs, which is also what a request that names no provider gets.
        return _resolve_elevenlabs(doc, "elevenlabs", model, descriptor, legacy)
    if provider == "openai":
        return _resolve_openai(doc, provider, model, descriptor, legacy)
    raise ValueError(f"speech voicing map has no provider {provider!r}")


def format_degraded(tokens: Sequence[str]) -> str:
    """The degraded tokens as the header value, capped so a header never fails a request.

    The cap is on bytes because that is what a header is measured in; the
    ellipsis is INSIDE the budget so the documented cap is what the header
    carries, and the cut is on a rune boundary so the value stays decodable.
    """
    max_bytes = 512
    value = ",".join(tokens)
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    budget = max_bytes - len("…".encode("utf-8"))
    encoded = value.encode("utf-8")[:budget]
    trimmed = encoded.decode("utf-8", errors="ignore")
    index = trimmed.rfind(",")
    if index > 0:
        trimmed = trimmed[:index]
    return trimmed + "…"


def degraded_tokens(attempt: Attempt) -> list[str]:
    """Convenience alias kept beside ``format_degraded`` for route callers."""
    return attempt.degraded_tokens()
