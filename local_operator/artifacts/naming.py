"""Display names for generated media, derived from the REQUEST.

**Why the request names the artifact.** Every generation lane used to name its
result from the provider's *model* id, so a Library full of generations read
``openai-gpt-image-2-5-sunburst-edit-01.png``: the same template for every
request, and two generations of different things under one name. The request
(prompt, or an agent-supplied ``filename`` stem) is the only thing that knows
what the asset *is*, so it is the naming input; a model id is provenance and
belongs in the caption/details, not in the name.

**Why the sanitiser is strict.** ``suggested`` is model-supplied text and an
agent can be steered by anything it just read, so it is hostile input: a name
is a label, never a path. The value is reduced to its last path component on
both separators, no separator or NUL survives, and ``.``/``..``/all-punctuation
fall through to the prompt. The destination joins this name onto a real
directory, so this module is the boundary that keeps ``../../etc/passwd`` a
*label* (``passwd``) rather than a traversal.

**Why the extension follows the bytes.** The extension is computed from the
asset's ``content_type`` — what the provider actually returned — never from
the suggested name: an agent asking for ``clip.mov`` does not make the bytes
QuickTime, and a browser/player keys its decoder off the suffix. The
preferred map pins the spellings the lanes' surfaces expect (``.jpg`` not
``.jpe``, ``.mp3`` for both ``audio/mpeg`` and ``audio/mp3``); ``mimetypes``
is the stdlib fallback, and the kind's conventional suffix is the last resort
for an unlabelled response.

**What this module deliberately does NOT do.** It does not deduplicate.
Uniqueness belongs to the real destination (the session's
``scratchpad/generated/``), which compares BYTES: identical bytes reuse the
existing file, different bytes with the same display name get a digest
suffix. A counter here would make two different prompts collide less but two
identical re-runs collide more, and it could not see the destination anyway.
The only counter is ``index``: one call returning several assets (a batch, or
``num_images > 1``) numbers them ``-2``, ``-3``, … while a single asset never
carries one.

This module is stdlib-only, like the rest of ``local_operator.artifacts``: the
session-build paths import artifact names and must not drag in pydantic or
HTTP clients.
"""

from __future__ import annotations

import mimetypes
import re

__all__ = ["media_filename"]

#: Preferred extensions per content type. The lanes' surfaces and stores key
#: off these spellings, and ``mimetypes`` alone disagrees on some of them
#: (``image/jpeg`` historically guesses ``.jpe``; ``audio/mp3`` guesses
#: nothing), so the map wins before the stdlib lookup.
_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}

#: The last-resort suffix when the provider labelled nothing usable: the
#: convention each artifact kind already had. Unknown kinds get ``.bin``,
#: which is honest about bytes nobody named.
_KIND_EXTENSIONS = {"image": ".png", "audio": ".mp3", "video": ".mp4"}
_UNKNOWN_EXTENSION = ".bin"

#: A filename stem long enough to describe the request, short enough to read
#: and to stay well under every filesystem's component limit once the
#: destination adds a digest suffix.
_MAX_STEM_CHARS = 60

#: A leading English article is noise in a filename ("A red fox…" is a red
#: fox). It is dropped ONLY when a separator follows, so "Anvil" and "Theory"
#: keep their first letters.
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)[\s,;:.!?_-]+", re.IGNORECASE)


def _slug(text: str) -> str:
    """Lowercase alnum runs joined by single ``-``; ``""`` when nothing is left.

    Unicode letters/digits survive (``.casefold()`` then ``isalnum``), which
    keeps a Japanese or accented prompt recognisable; everything else — path
    separators, NUL, punctuation, underscores — collapses to the separator.
    """
    folded = _LEADING_ARTICLE.sub("", text.casefold(), count=1)
    joined = "".join(ch if ch.isalnum() else "-" for ch in folded)
    return re.sub(r"-+", "-", joined).strip("-")


def _from_suggestion(suggested: str | None) -> str:
    """The usable stem a suggested name yields, or ``""`` to fall through.

    Hostile input: only the last component of a path survives (both
    separators), then a trailing extension is dropped because the extension
    comes from the bytes, then leading dots/separators (``.hidden``, ``..``)
    are stripped. Anything that leaves an empty stem — ``.``, ``..``, ``///``,
    all punctuation — is not a name and must fall through to the prompt.
    """
    if not suggested:
        return ""
    candidate = re.split(r"[/\\]", suggested.strip())[-1]
    stem, dot, extension = candidate.rpartition(".")
    if dot and extension and extension.isalnum() and len(extension) <= 10:
        candidate = stem
    candidate = candidate.lstrip(". -_")
    return _slug(candidate)


def _cap(stem: str) -> str:
    """Cut at the last word boundary inside the cap; hard-cut a single word.

    Word boundaries keep the name readable ("…-golden-hour" rather than
    "…-golden-ho"); a single 100-character word has no boundary to honour, so
    the hard cut is the only honest option left.
    """
    if len(stem) <= _MAX_STEM_CHARS:
        return stem
    head = stem[:_MAX_STEM_CHARS]
    boundary = head.rfind("-")
    return head[:boundary] if boundary > 0 else head


def _extension(content_type: str | None, kind: str) -> str:
    """The suffix for the provider's bytes: preferred map, stdlib, kind."""
    base = (content_type or "").split(";", 1)[0].strip().lower()
    if base in _CONTENT_TYPE_EXTENSIONS:
        return _CONTENT_TYPE_EXTENSIONS[base]
    guessed = mimetypes.guess_extension(base) if base else None
    if guessed:
        return guessed
    return _KIND_EXTENSIONS.get(kind, _UNKNOWN_EXTENSION)


def media_filename(
    prompt: str | None,
    content_type: str | None,
    *,
    suggested: str | None = None,
    kind: str = "media",
    index: int = 1,
) -> str:
    """``red-fox-in-snow.png`` — the display name for one generated asset.

    Preference order: the agent-supplied ``suggested`` stem, else the prompt,
    else the kind word (``image``/``audio``/``video``/``media``). The extension
    always comes from ``content_type``, and ``index >= 2`` numbers assets that
    one call returned together.
    """
    stem = _from_suggestion(suggested) or _slug(prompt or "") or kind
    stem = _cap(stem)
    suffix = f"-{index}" if index >= 2 else ""
    return f"{stem}{suffix}{_extension(content_type, kind)}"
