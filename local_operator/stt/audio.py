"""Audio caps and format helpers for the speech cascade.

Constants only — v1 adds NO configuration keys (design §3c), so nothing here
reads settings; if that ever changes, the key must be registered in
``settings_io.py`` first (AGENTS.md, "Adding a configuration key").

The numbers come from the design's §4d: the binding budgets are the desktop
body cap (900_000 bytes of JSON, and base64 inflates 4/3) and the runtime
control-socket line cap (1 MiB). :data:`MAX_AUDIO_BYTES` is set so a capture at
this cap still fits the desktop body with room for the rest of the payload.
Duration is a capture-side guide — the daemon cannot measure it without a
decoder and does not try.
"""

from __future__ import annotations

import os

from local_operator import media
from local_operator.media import AudioInfo

#: Longest capture the surfaces should produce, in seconds. Not enforced
#: server-side (no decoder); it is the contract the client legs capture to.
MAX_AUDIO_DURATION_S = 60

#: Raw audio byte cap. 60 s at 64 kbps mono is exactly this; base64 then costs
#: ~640 kB of the 900 kB desktop body, which is the budget that actually binds.
MAX_AUDIO_BYTES = 480_000

#: Mime allowlist for captured audio (§4b). Declared mime is never trusted at
#: admission — phase 1 sniffs the bytes — but every surface validates against
#: this set so a typo'd container is refused by name.
AUDIO_MIME_TYPES = frozenset(
    {
        "audio/wav",
        "audio/mpeg",
        "audio/mp4",
        "audio/webm",
        "audio/ogg",
    }
)

#: Mime → the ``input_audio.format`` token the OpenAI-compatible chat wire
#: takes. **Only the model-audio rung has this restriction** (OQ-3): the STT
#: rungs pass webm/opus through fine. A format outside this map must be
#: refused at admission for the audio door — v1 does not transcode.
MODEL_AUDIO_WIRE_FORMATS = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
}

#: Mimes Gemini's inline audio part documents for audio input, fetched
#: 2026-09-29 from https://ai.google.dev/gemini-api/docs/audio ("Supported
#: audio formats", page updated 2026-09-23) — the design §4b list was wider in
#: the live page than in the draft, which is why the review round that added
#: the admission gate asked for the list to be re-verified against the vendor.
#: The entries are the vendor's OWN spellings, kept verbatim: the admission
#: gate mirrors the documented page rather than guessing family aliases, so
#: ``audio/mp4`` — our sniffer's report for the M4A container, which the page
#: spells only ``audio/m4a`` — stays refused on this wire while the STT rung
#: carries it happily (v1 renames no declarations and transcodes nothing).
GOOGLE_MODEL_AUDIO_MIME_TYPES = frozenset(
    {
        "audio/wav",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/mpeg",
        "audio/m4a",
        "audio/l16",
        "audio/opus",
        "audio/alaw",
        "audio/mulaw",
        "audio/webm",
    }
)

#: Suffix → mime, for the temp file the route writes from an upload. Covers the
#: allowlist plus the common container spellings for each, plus the two formats
#: ``media.sniff_audio`` can report beyond the allowlist (flac, aiff): they are
#: accepted by both STT providers, so the upload must name them truthfully.
_SUFFIX_MIME = {
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
}

#: Mime → upload suffix, the inverse of :data:`_SUFFIX_MIME` for the allowlist
#: plus the sniffer's extra formats.
_MIME_SUFFIX = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/aiff": ".aiff",
}


def mime_for_path(path: str | os.PathLike[str]) -> str:
    """The mime a saved capture path implies, from its suffix.

    Falls back to ``audio/wav`` for an unknown suffix: every route in this tree
    names the temp file from the upload's own content type, so the fallback is
    only reached by a direct caller, and the client legs capture wav by default.
    """
    suffix = os.path.splitext(str(path))[1].lower()
    return _SUFFIX_MIME.get(suffix, "audio/wav")


def extension_for_mime(mime: str) -> str:
    """The upload filename suffix for ``mime``.

    Unknown mimes get ``.bin`` rather than a guessed container: both STT
    providers key decoding off the part's filename, so naming an ADTS stream
    ``.wav`` would have them decode it as the wrong format — a refusal is the
    honest answer, and the executor falls forward on it.
    """
    return _MIME_SUFFIX.get(mime, ".bin")


def filename_for_mime(mime: str) -> str:
    """A multipart filename matching ``mime``.

    Both STT providers key their decoding off the part's filename/content type,
    so the name is derived from the mime rather than invented per call site.
    """
    return f"audio{extension_for_mime(mime)}"


def format_for_model_wire(mime: str) -> str | None:
    """The ``input_audio`` format token for ``mime``, or ``None`` if the wire refuses it.

    ``None`` is the honest-answer half of OQ-3: v1 sends only what the target
    model's wire takes, and a caller that gets ``None`` must report the
    model-audio rung unavailable rather than transcode.
    """
    return MODEL_AUDIO_WIRE_FORMATS.get(mime)


def sniff_audio(data: bytes) -> AudioInfo | None:
    """Sniff a captured container from its header bytes.

    Delegates to :func:`local_operator.media.sniff_audio` — the ONE
    implementation (a second header walk here is how two sniffers drift).
    ``None`` means "do not treat this as audio"; callers on the capture path
    must treat it exactly like a failed sniff (v1 admission refuses what it
    cannot name), while the STT executor may fall back to the file's suffix —
    its upload is first-party, not an untrusted block.
    """
    return media.sniff_audio(data)
