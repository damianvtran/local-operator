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
  removed. Its rules are deliberately small: only languages where the candidate
  token is not a real word, a closed non-speech cue lexicon, and a hard no-op
  when the language is unknown.

The mode is a closed vocabulary. An unknown or not-yet-implemented mode is a
caller error (:class:`ValueError`), never a silent fallback to another cleanup:
a user who asked for verbatim must not receive a cleaned transcript, and vice
versa. ``auto``, ``provider``, ``client`` and ``off`` are implemented;
``edit`` remains recognized-but-rejected until its opt-in slice.

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

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from local_operator.stt import RemovedSpan

__all__ = [
    "CLEANUP_MODES",
    "CLEANUP_SETTING_KEY",
    "DEFAULT_CLEANUP",
    "RAW_UNAVAILABLE_PROVIDER_NO_VERBATIM",
    "SUPPORTED_CLEANUP_MODES",
    "CleanupMode",
    "ClientRulesResult",
    "ISO_639_1",
    "InvalidCleanupMode",
    "ProviderCleanupOptions",
    "RemovedSpan",
    "apply_client_rules",
    "choose_cleanup_language",
    "configured_cleanup",
    "elevenlabs_cleanup_options",
    "normalize_language_hint",
    "resolve_cleanup",
]

#: The closed vocabulary a request may name. ``client`` and ``edit`` are
#: reserved here so the wire spelling is stable before their behaviour lands.
CleanupMode = Literal["auto", "provider", "client", "off", "edit"]

CLEANUP_MODES: tuple[CleanupMode, ...] = ("auto", "provider", "client", "off", "edit")

#: Modes this build can execute. ``edit`` is recognized and refused until its
#: opt-in slice; every other mode has a defined behaviour now.
SUPPORTED_CLEANUP_MODES: tuple[CleanupMode, ...] = ("auto", "provider", "client", "off")

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
    Provenance is not stored here: the client that actually put a field on the
    wire reports it, so a fake/older client cannot claim a cleanup it never
    asked for.
    """

    no_verbatim: bool | None = None
    tag_audio_events: bool | None = None


class InvalidCleanupMode(ValueError):
    """A caller named a mode this build cannot execute.

    A distinct type (still a ValueError for existing callers) so the desktop
    route can answer 422 instead of inheriting the pre-existing prompt-length
    400 arm. The mode is never silently downgraded: ``edit`` becoming
    ``provider`` would spend a different amount and return a different
    artifact.
    """


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
        raise InvalidCleanupMode(
            f"Unknown transcription cleanup mode {value!r}. " f"Expected one of: {allowed}."
        )
    if mode not in SUPPORTED_CLEANUP_MODES:
        raise InvalidCleanupMode(
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
        return ProviderCleanupOptions(no_verbatim=True, tag_audio_events=False)
    if mode == "client":
        # The client rules need the provider's text BEFORE cleaning. Sending
        # ``no_verbatim=false`` explicitly keeps the provider from inheriting a
        # default, and ``tag_audio_events=false`` removes the cue labels the
        # closed lexicon would otherwise have to strip after the fact.
        return ProviderCleanupOptions(no_verbatim=False, tag_audio_events=False)
    if mode == "off":
        return ProviderCleanupOptions()
    # ``edit`` is rejected by resolve_cleanup before reaching here; refusing
    # rather than answering a plausible-looking option keeps a future mode from
    # silently behaving like a different one.
    raise InvalidCleanupMode(f"ElevenLabs cleanup mode {mode!r} is not supported.")


# ---------------------------------------------------------------------------
# Client rules v1
# ---------------------------------------------------------------------------

#: ISO-639-1 is the only value Radient's ``language`` field accepts (a live
#: ``en-US`` or a 3-letter ``fil`` is rejected with a 400). This explicit
#: allowlist — not a shape check — is what Q21 asked for: an unmapped 2-letter
#: spelling must be omitted, never forwarded on the chance that it is valid.
ISO_639_1: frozenset[str] = frozenset("""
    aa ab ae af ak am an ar as av ay az ba be bg bi bm bn bo br bs ca ce ch co
    cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd
    gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv
    ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg
    mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os
    pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss
    st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo
    wa wo xh yi yo za zh zu
    """.split())

#: The provider reports ISO-639-3 (``eng``, ``por``) while the rule table and
#: Radient speak ISO-639-1. Only the languages this workstream has evidence for
#: are mapped; an unmapped 3-letter label is unknown, not guessed.
ISO_639_3_TO_1: dict[str, str] = {
    "afr": "af",
    "amh": "am",
    "ara": "ar",
    "asm": "as",
    "aze": "az",
    "bel": "be",
    "ben": "bn",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cmn": "zh",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "eus": "eu",
    "fas": "fa",
    "fin": "fi",
    "fra": "fr",
    "glg": "gl",
    "guj": "gu",
    "heb": "he",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "ind": "id",
    "ita": "it",
    "jpn": "ja",
    "kan": "kn",
    "kaz": "kk",
    "khm": "km",
    "kor": "ko",
    "lao": "lo",
    "mal": "ml",
    "mar": "mr",
    "msa": "ms",
    "mya": "my",
    "nep": "ne",
    "nld": "nl",
    "nor": "no",
    "pan": "pa",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "sin": "si",
    "slk": "sk",
    "slv": "sl",
    "som": "so",
    "spa": "es",
    "srp": "sr",
    "swa": "sw",
    "swe": "sv",
    "tam": "ta",
    "tel": "te",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vie": "vi",
    "yor": "yo",
    "zho": "zh",
    "zul": "zu",
}

#: A detected-language label below this confidence is not trusted for token
#: rules. The live Hindi probe returned ``hin`` once and ``por`` twice for the
#: same voice, so a bare label is not enough.
LANGUAGE_PROBABILITY_THRESHOLD = 0.8

#: Provenance token for a text cleaned by this module's rules.
CLIENT_RULES_PROVENANCE = "client.rules.v1"

#: Token sets. ``um`` is deliberately absent from English: it is a real
#: Portuguese numeral/article, and one language label cannot tell an English
#: filler from Portuguese ``Um livro`` inside a code-switched transcript.
TOKEN_RULES: dict[str, tuple[str, ...]] = {
    "en": ("uh", "uhm", "erm"),
    "fr": ("euh",),
}

#: Closed non-speech cue lexicon. Deliberately NOT generic bracket removal:
#: ``[inaudible]``, ``[sic]``, ``[REDACTED]`` and entity labels are information.
CUE_LABELS: frozenset[str] = frozenset(
    {
        "applause",
        "breath",
        "breathing",
        "clears throat",
        "coughing",
        "coughs",
        "footsteps",
        "laughter",
        "laughing",
        "laughs",
        "lip smack",
        "lip smacking",
        "music",
        "noise",
        "pause",
        "sighing",
        "sighs",
        "silence",
        "sniffles",
        "sniffs",
        "static",
        "throat clearing",
        "whistles",
    }
)

_CUE_RE = re.compile(r"[\[\(]\s*([^\[\]\(\)]{1,40}?)\s*[\]\)]", re.IGNORECASE)
_WORD_CATEGORIES = frozenset({"L", "M", "N"})
_CONNECTOR_CATEGORY = "Pc"
_JOINERS = frozenset({"\u200c", "\u200d"})


@dataclass(frozen=True)
class ClientRulesResult:
    """The result of one client-rule pass over a provider transcript."""

    text: str
    raw_text: str
    removed: tuple[RemovedSpan, ...] = ()
    cleanup: str = CLIENT_RULES_PROVENANCE


def normalize_language_hint(value: str | None) -> str | None:
    """Map a BCP-47/ISO-639-3 hint to an allowed ISO-639-1 code, or ``None``.

    ``en-US`` -> ``en``, ``zh-Hans`` -> ``zh``, ``eng`` -> ``en``. A primary
    subtag that is not in :data:`ISO_639_1` (``fil-PH`` -> ``fil``, ``xx``)
    answers ``None`` so the caller omits the provider field and the token rules
    no-op. This is the Q20/Q21 guard: the wire accepts only the allowlist, not
    every two-letter spelling.
    """
    if value is None:
        return None
    raw = str(value).strip().replace("_", "-")
    if not raw:
        return None
    primary = raw.split("-", 1)[0].lower()
    if len(primary) == 3:
        primary = ISO_639_3_TO_1.get(primary, primary)
    return primary if primary in ISO_639_1 else None


def _is_word_continuation(char: str) -> bool:
    """True when ``char`` makes a token part of a larger word.

    Uses Unicode categories rather than ``\\w``: decomposed accents (a base
    letter + combining mark), ZWJ/ZWNJ joins and connector punctuation stay
    word-internal; CJK, Arabic and RTL letters are covered by ``L*``/``M*``.
    No normalization is performed.
    """
    if char in _JOINERS or char == "-":
        return True
    category = unicodedata.category(char)
    return category[0] in _WORD_CATEGORIES or category == _CONNECTOR_CATEGORY


def _boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and _is_word_continuation(text[start - 1]):
        return False
    if end < len(text) and _is_word_continuation(text[end]):
        return False
    return True


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Code/quote spans a token must not be removed from."""
    spans: list[tuple[int, int]] = []
    for pattern in (r"```.*?```", r"`[^`\n]*`", r'"[^"\n]*"', r"“[^”\n]*”"):
        spans.extend(match.span() for match in re.finditer(pattern, text, re.DOTALL))
    return spans


def _inside(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _cue_spans(text: str) -> list[tuple[int, int, str, str]]:
    spans: list[tuple[int, int, str, str]] = []
    for match in _CUE_RE.finditer(text):
        label = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        if label in CUE_LABELS:
            spans.append((match.start(), match.end(), "cue", match.group(0)))
    return spans


def _expand_deletion(text: str, start: int, end: int, kind: str) -> list[tuple[int, int]]:
    """Expand one removed span over only the separators that existed for it.

    Fillers take one adjacent separator (``Uh, so`` -> ``so``; ``I, uh, think``
    -> ``I think``; ``Okay. Uh. Let's`` -> ``Okay. Let's``). Cues take the
    following space when present, else the preceding one, so two words never
    join without a space.
    """
    if kind == "cue":
        if end < len(text) and text[end] == " ":
            return [(start, end + 1)]
        if start > 0 and text[start - 1] == " ":
            return [(start - 1, end)]
        return [(start, end)]

    left_comma = start >= 2 and text[start - 2 : start] == ", "
    right_comma = text[end : end + 2] == ", "
    if left_comma and right_comma:
        # ``I, uh, think`` -> ``I think``: remove the two commas but keep the
        # space that separated ``I`` from the first comma. Removing one
        # contiguous range would join the words as ``Ithink``.
        return [(start - 2, start - 1), (start, end + 2)]
    if end + 1 < len(text) and text[end] in ",.;:!?" and text[end + 1] == " ":
        return [(start, end + 2)]
    if end < len(text) and text[end] == " ":
        return [(start, end + 1)]
    if left_comma:
        return [(start - 2, end)]
    if start > 0 and text[start - 1] == " ":
        return [(start - 1, end)]
    return [(start, end)]


def _merge_deletions(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_sentence_initial(text: str, start: int) -> bool:
    prefix = text[:start].rstrip(" \t\n")
    return not prefix or prefix[-1] in ".!?"


def _apply_sentence_case(
    output: str,
    text: str,
    fillers: list[tuple[int, int]],
    deletions: list[tuple[int, int]],
) -> str:
    """Upper-case the first letter after a sentence-initial filler removal.

    Only that one letter changes; this is the sole non-deletion edit in the
    rules and exists so ``Uh, so ...`` becomes ``So ...``, not ``so ...``.
    """
    for start, end in fillers:
        if not _is_sentence_initial(text, start):
            continue
        # The expansion around this filler, not the token itself, is where the
        # next word begins in the output.
        expanded = next(
            (d for d in deletions if d[0] <= start and d[1] >= end),
            (start, end),
        )
        removed_before = sum(
            d_end - d_start for d_start, d_end in deletions if d_end <= expanded[0]
        )
        index = expanded[0] - removed_before
        while index < len(output) and output[index] in " \t\n":
            index += 1
        if index < len(output) and output[index].islower():
            output = output[:index] + output[index].upper() + output[index + 1 :]
    return output


def apply_client_rules(text: str, *, language: str | None = None) -> ClientRulesResult:
    """Remove v1 fillers and non-speech cues without touching real words.

    ``language`` must already be a normalised ISO-639-1 code (or ``None``); the
    callers apply :func:`normalize_language_hint` and the probability gate.
    Unknown language = token rules no-op, cue removal still runs. The input is
    returned unchanged as ``raw_text``; ``removed`` carries audit spans in raw
    coordinates. No Unicode normalisation or case folding is applied to the
    surviving text.
    """
    protected = _protected_spans(text)
    cues = [cue for cue in _cue_spans(text) if not _inside(protected, cue[0], cue[1])]
    cue_ranges = [(cue_start, cue_end) for cue_start, cue_end, _, _ in cues]
    spans: list[tuple[int, int, str, str]] = list(cues)

    for token in TOKEN_RULES.get(language or "", ()):
        for match in re.finditer(re.escape(token), text, re.IGNORECASE):
            start, end = match.span()
            matched = match.group(0)
            # Abbreviations survive: UM/UH are not fillers.
            if len(matched) > 1 and matched.isupper():
                continue
            if not _boundary_ok(text, start, end):
                continue
            if _inside(protected, start, end) or _inside(cue_ranges, start, end):
                continue
            spans.append((start, end, "filler", matched))

    if not spans:
        return ClientRulesResult(text=text, raw_text=text)

    expanded: list[tuple[int, int]] = []
    for start, end, kind, _ in spans:
        expanded.extend(_expand_deletion(text, start, end, kind))
    deletions = _merge_deletions(expanded)

    pieces: list[str] = []
    cursor = 0
    for start, end in deletions:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    output = "".join(pieces)

    fillers = [(start, end) for start, end, kind, _ in spans if kind == "filler"]
    output = _apply_sentence_case(output, text, fillers, deletions)

    removed = tuple(
        RemovedSpan(start=start, end=end, kind=kind, token=token)
        for start, end, kind, token in sorted(spans)
    )
    return ClientRulesResult(text=output, raw_text=text, removed=removed)


def choose_cleanup_language(
    caller: str | None, detected: str | None, probability: float | None
) -> str | None:
    """The one language decision for client rules.

    A caller-supplied value is authoritative — including when it is not an
    ISO-639-1 code (``fil-PH`` -> unknown, not a guessed ``fil``), because
    cleaning a Tagalog take with English rules is the failure direction this
    design refuses. Only an absent caller hint falls back to the provider's
    label, and only above the confidence threshold.
    """
    if caller is not None and str(caller).strip():
        return normalize_language_hint(caller)
    if probability is not None and probability >= LANGUAGE_PROBABILITY_THRESHOLD:
        return normalize_language_hint(detected)
    return None


#: The `/settings` key this policy reads. One contract: the registry row, the
#: TUI/UI page, the routes and the docs all name this key.
CLEANUP_SETTING_KEY = "speech.transcription.cleanup"


def configured_cleanup(manager: Any) -> CleanupMode:
    """The configured cleanup mode, or ``auto`` when it cannot be read.

    The routes call this ONLY when the request names no mode; a per-call
    ``cleanup`` field always wins. A hand-edited invalid value falls back to
    ``auto`` rather than crashing a transcription — writes through
    ``settings_io`` are rejected before persistence, so this is the
    unreadable/legacy-file rung, not the normal one.
    """
    from local_operator import settings_io

    setting = settings_io.BY_KEY.get(CLEANUP_SETTING_KEY)
    if setting is None:  # a tree without the row (older plugin/consumer)
        return DEFAULT_CLEANUP
    try:
        return resolve_cleanup(settings_io.read_setting(manager, setting))
    except Exception:  # noqa: BLE001 — an unreadable/invalid setting must not fail dictation
        return DEFAULT_CLEANUP
