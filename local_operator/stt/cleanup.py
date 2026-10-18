"""Transcript cleanup policy: the mode vocabulary and the provider request fields.

WHY THIS MODULE EXISTS. Cleanup has two places it can act, and they are not
interchangeable:

* **Provider-native** — the vendor's own model removes the tokens while it
  transcribes (ElevenLabs Scribe v2's ``no_verbatim``). This is the best lever:
  it is language-aware and free, but it is also destructive at the provider, so
  no raw copy comes back. The module owns the *request fields* that switch it on.
* **Client-side** — a deterministic post-pass over the text. It is the only
  lever for rungs whose provider has no native switch (Radient, OpenAI), and the
  only one that can hand back both the cleaned text and the exact spans it
  removed. The rules themselves land in a later slice; the mode vocabulary and
  provenance are frozen here so the wire shape does not move twice.

The mode is a closed vocabulary. An unknown or not-yet-implemented mode is a
caller error (:class:`ValueError`), never a silent fallback to another cleanup:
a user who asked for verbatim must not receive a cleaned transcript, and vice
versa. Slice 2 implements ``auto``, ``provider`` and ``off``. ``client`` and
``edit`` are recognized so the wire can name them, but reject until their
slices land (client rules, then the opt-in surcharged edit).

RAW PRESERVATION INVARIANT. Our layer never destroys the only copy it holds.
Client rules must return their input as ``raw_text``; ``off`` returns the
provider text unchanged. Provider-native cleanup is different by contract: the
provider removes the tokens before returning, so ``raw_text`` is absent and
``raw_unavailable`` says exactly why (``provider_no_verbatim``). A caller must
distinguish that from "nothing was removed" (``raw_text == text``). No surface
may offer an "original transcript" view when ``raw_unavailable`` is set.

LANGUAGE LIMITATION, stated where the switch is flipped: Scribe v2's
``no_verbatim`` was measured on 2026-10-18 to remove English/French fillers,
preserve Portuguese ``um`` and Arabic ``أمي`` (real words), and LEAVE Japanese
``えーと``, Chinese ``嗯`` and Hindi ``अरे``. Under-cleaning is the safe failure
direction; the Radient rung's client rules (slice 3) and the opt-in
``transcript_edit`` mode (slice 5) cover the remaining languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from local_operator.stt import RemovedSpan

__all__ = [
    "CLEANUP_MODES",
    "DEFAULT_CLEANUP",
    "RAW_UNAVAILABLE_PROVIDER_NO_VERBATIM",
    "SUPPORTED_CLEANUP_MODES",
    "CleanupMode",
    "ProviderCleanupOptions",
    "RemovedSpan",
    "elevenlabs_cleanup_options",
    "resolve_cleanup",
]

#: The closed vocabulary a request may name. ``client`` and ``edit`` are
#: reserved here so the wire spelling is stable before their behaviour lands.
CleanupMode = Literal["auto", "provider", "client", "off", "edit"]

CLEANUP_MODES: tuple[CleanupMode, ...] = ("auto", "provider", "client", "off", "edit")

#: Modes this slice can actually execute. The rest are recognized and refused
#: with a typed ValueError (the routes map that to 422).
SUPPORTED_CLEANUP_MODES: tuple[CleanupMode, ...] = ("auto", "provider", "off")

#: Default when a caller names no mode. ``auto`` is the only default that uses
#: a provider's native switch when one exists.
DEFAULT_CLEANUP: CleanupMode = "auto"

#: Why a provider-native cleanup leaves no raw copy. This is a wire value:
#: consumers compare it instead of guessing from a ``None``.
RAW_UNAVAILABLE_PROVIDER_NO_VERBATIM = "provider_no_verbatim"


@dataclass(frozen=True)
class ProviderCleanupOptions:
    """The request fields one provider needs for a cleanup mode.

    ``None`` means "do not send the field"; the provider's own default applies.
    ``cleanup`` is the provenance token the served transcript carries (or
    ``"none"``), and ``raw_unavailable`` states why no raw copy can exist.
    """

    no_verbatim: bool | None = None
    tag_audio_events: bool | None = None
    cleanup: str = "none"
    raw_unavailable: str = ""


def resolve_cleanup(value: str | None) -> CleanupMode:
    """Validate a request's cleanup mode, or answer the default.

    Empty/absent is ``auto``. An unknown spelling, or a recognized-but-not-yet-
    implemented mode, raises ValueError with a sentence a 422 can carry. The
    mode is deliberately NOT silently downgraded: "edit" becoming "provider"
    would spend a different amount of money and return a different artifact
    than the caller asked for.
    """
    if value is None or not str(value).strip():
        return DEFAULT_CLEANUP
    mode = str(value).strip().lower()
    if mode not in CLEANUP_MODES:
        allowed = ", ".join(CLEANUP_MODES)
        raise ValueError(
            f"Unknown transcription cleanup mode {value!r}. " f"Expected one of: {allowed}."
        )
    if mode not in SUPPORTED_CLEANUP_MODES:
        raise ValueError(
            f"Transcription cleanup mode {mode!r} is not available yet. "
            f"Use one of: {', '.join(SUPPORTED_CLEANUP_MODES)}."
        )
    return mode  # type: ignore[return-value]


def elevenlabs_cleanup_options(mode: CleanupMode) -> ProviderCleanupOptions:
    """The Scribe v2 request fields for ``mode``.

    ``auto``/``provider`` turn on the provider's native no-verbatim switch and
    explicitly suppress audio-event tags (the field has no documented default,
    so it is never left implicit). ``off`` sends NEITHER field: the request must
    stay byte-identical to the pre-cleanup client for a verbatim caller.
    """
    if mode in ("auto", "provider"):
        return ProviderCleanupOptions(
            no_verbatim=True,
            tag_audio_events=False,
            cleanup="elevenlabs.no_verbatim",
            raw_unavailable=RAW_UNAVAILABLE_PROVIDER_NO_VERBATIM,
        )
    if mode == "off":
        return ProviderCleanupOptions()
    # ``client`` and ``edit`` are rejected by resolve_cleanup before reaching
    # here; refusing rather than answering a plausible-looking option keeps a
    # future mode from silently behaving like a different one.
    raise ValueError(f"ElevenLabs cleanup mode {mode!r} is not supported.")
