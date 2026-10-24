"""Provider-neutral Radient cleanup options and response normalization.

The legacy desktop route, cascade rung and phone bridge all use this module so
an older relay without native metadata remains compatible, while native Scribe
cleanup never suppresses the deterministic CJK backstop in ``auto`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from local_operator.stt.cleanup import (
    CLIENT_RULES_PROVENANCE,
    RemovedSpan,
    apply_client_rules,
    choose_cleanup_language,
    normalize_language_hint,
    resolve_cleanup,
)

_LANGUAGE_PROMPT_GUIDANCE = {
    "en": (
        "The audio is English; preserve that language. Omit hesitation sounds such as "
        "um, uh, er and erm; preserve meaningful ah/hmm backchannels and mm measurements."
    ),
    "de": (
        "The audio is German; preserve that language. Omit äh or ähm only as hesitations; "
        "preserve äh when it expresses disgust."
    ),
    "fr": (
        "The audio is French; preserve that language. Omit hesitation euh; preserve ben "
        "when it marks assent or surprise."
    ),
    "it": "The audio is Italian; preserve that language. Omit hesitation ehm only.",
    "es": (
        "The audio is Spanish; preserve that language. Omit eh or este only as clear "
        "hesitations; preserve demonstrative este, as in Toma este libro."
    ),
    "pt": (
        "The audio is Portuguese; preserve that language. Omit only clear hesitations; "
        "preserve Um as an article and hã as a question or response."
    ),
    "ja": (
        "The audio is Japanese; preserve that language. Omit hesitation えーと, ええと or "
        "あの only as a hesitation; preserve demonstrative あの."
    ),
    "zh": (
        "The audio is Chinese; preserve that language. Omit clear hesitation 呃; preserve "
        "assent 嗯 and real words such as 呃逆."
    ),
    "hi": (
        "The audio is Hindi; preserve that language. Omit उम्म only as a clear hesitation; "
        "preserve it when used as a real noun."
    ),
    "ru": (
        "The audio is Russian; preserve that language. Omit э or ну only as clear "
        "hesitations; preserve objection э and urging ну."
    ),
    "he": "The audio is Hebrew; preserve that language. Omit hesitation אממ only.",
}


@dataclass(frozen=True)
class RadientCleanupResult:
    """One seam-independent transcription result after local normalization.

    Removed spans refer to the provider-returned text before the local pass.
    When native no-verbatim cleanup ran, that text is not a verbatim raw copy;
    ``raw_unavailable`` preserves that distinction instead of inventing one.
    """

    text: str
    raw_text: str | None
    raw_unavailable: str
    language: str | None
    language_probability: float | None
    cleanup: str
    removed: tuple[RemovedSpan, ...]


def stt_cleanup_prompt(
    mode: str,
    prompt: str | None,
    language: str | None,
) -> str | None:
    """Return bounded STT-only guidance only when the caller supplied no prompt.

    Caller prompts are opaque context and are never replaced or truncated. The
    generated instruction stays below Radient's 1000-character wire ceiling.
    """
    resolved = resolve_cleanup(mode)
    if prompt is not None or resolved not in ("auto", "provider"):
        return prompt
    normalized = normalize_language_hint(language)
    language_clause = _LANGUAGE_PROMPT_GUIDANCE.get(
        normalized or "",
        "Preserve the audio's language, including code switching. Omit only clear "
        "hesitation sounds; preserve assent, backchannels, demonstratives and real words.",
    )
    return (
        "Transcribe the audio faithfully. Omit filler words, hesitations and "
        "non-speech event annotations, but preserve meaning, word order, names, "
        f"numbers and all substantive content. {language_clause} "
        "Do not rewrite, summarize, translate or add commentary."
    )


def radient_cleanup_options(
    mode: str,
    prompt: str | None,
    language: str | None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Map the shared cleanup vocabulary to optional Radient multipart fields."""
    resolved = resolve_cleanup(mode)
    remove: bool | None = None
    include_events: bool | None = None
    # Native cleanup is ElevenLabs-only. An explicitly different provider must
    # not receive controls the relay correctly rejects; an omitted provider
    # keeps the default-Radient native path and its ElevenLabs fix unchanged.
    explicit_non_elevenlabs = (
        isinstance(provider, str)
        and bool(provider.strip())
        and provider.strip().lower() != "elevenlabs"
    )
    if not explicit_non_elevenlabs and resolved in ("auto", "provider"):
        remove, include_events = True, False
    elif not explicit_non_elevenlabs and resolved == "client":
        remove, include_events = False, False
    return {
        "remove_disfluencies": remove,
        "include_audio_events": include_events,
        "prompt": stt_cleanup_prompt(resolved, prompt, language),
    }


def normalize_radient_cleanup(
    result: Any,
    mode: str,
    *,
    caller_language: str | None = None,
) -> RadientCleanupResult:
    """Normalize new and legacy relay responses without inferring native success."""
    resolved = resolve_cleanup(mode)
    dump = getattr(result, "model_dump", None)
    payload = cast("dict[str, Any]", dump()) if callable(dump) else {}

    def field(name: str) -> Any:
        return getattr(result, name, payload.get(name))

    provider_text = str(field("text") or "")
    detected = field("language") or field("language_code")
    detected = detected if isinstance(detected, str) else None
    raw_probability = field("language_probability")
    probability = float(raw_probability) if isinstance(raw_probability, (int, float)) else None
    raw_unavailable = str(field("raw_unavailable") or "")
    reported_raw = field("raw_text")
    if reported_raw is not None:
        raw_text: str | None = str(reported_raw)
    elif raw_unavailable:
        raw_text = None
    else:
        # Legacy relays omitted both fields and returned the only available text;
        # that absence is not the newer provider_no_verbatim signal.
        raw_text = provider_text

    reported_cleanup = str(field("cleanup") or "")
    native_ack = ""
    if resolved in ("auto", "provider") and reported_cleanup not in (
        "",
        "none",
        "client.rules.v1",
        CLIENT_RULES_PROVENANCE,
    ):
        native_ack = reported_cleanup

    cleaned = None
    if resolved in ("auto", "client"):
        cleaned = apply_client_rules(
            provider_text,
            language=choose_cleanup_language(caller_language, detected, probability),
        )
    provenance = [native_ack] if native_ack else []
    if cleaned is not None:
        provenance.append(CLIENT_RULES_PROVENANCE)
    return RadientCleanupResult(
        text=cleaned.text if cleaned is not None else provider_text,
        raw_text=raw_text,
        raw_unavailable=raw_unavailable,
        language=detected,
        language_probability=probability,
        cleanup=".".join(provenance) if provenance else "none",
        removed=cleaned.removed if cleaned is not None else (),
    )
