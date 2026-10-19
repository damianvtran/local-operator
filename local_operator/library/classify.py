"""One pure classifier for kind/category/MIME, shared by every Library surface.

WHY ONE MODULE. The scanner records a ``kind`` from the Highlights contract's
grammar (``supplements.candidates.kind_of``) because that grammar decides which
files are previewable at all. The Library needs a wider answer — category
buckets, MIME for the content route, the office formats — and every surface
(desktop, TUI, phone, Expo, agent tool) must agree. A second copy in a route or
a renderer is how the same file ends up an ``image`` in one grid and an
``other`` in another.

PURITY. No filesystem, no clock, no imports beyond the stdlib. ``name`` may be
a basename or a path; only its suffix is read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

#: Preview kinds. The order here is also the filter-rail order surfaces use.
KINDS: Final = (
    "image",
    "audio",
    "video",
    "pdf",
    "pptx",
    "docx",
    "markdown",
    "html",
    "spreadsheet",
    "text",
    "code",
    "other",
)

#: Sidebar buckets. ``sound`` is the user-facing word for audio.
CATEGORIES: Final = (
    "image",
    "sound",
    "document",
    "presentation",
    "video",
    "spreadsheet",
    "code",
    "other",
)

_KIND_CATEGORY: Final = {
    "image": "image",
    "audio": "sound",
    "video": "video",
    "pdf": "document",
    "pptx": "presentation",
    "docx": "document",
    "markdown": "document",
    "html": "document",
    "spreadsheet": "spreadsheet",
    "text": "document",
    "code": "code",
    "other": "other",
}

#: Suffix -> kind. Deliberately explicit and small: an unknown suffix is
#: ``other`` rather than a guess, because a wrong preview promise is worse
#: than an honest "open externally".
_EXTENSIONS: Final = {
    # image
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".heic": "image",
    ".heif": "image",
    ".avif": "image",
    ".svg": "image",
    ".ico": "image",
    # audio
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".opus": "audio",
    ".aiff": "audio",
    ".aif": "audio",
    # video
    ".mp4": "video",
    ".mov": "video",
    ".m4v": "video",
    ".webm": "video",
    ".mkv": "video",
    ".avi": "video",
    # documents
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".docx": "docx",
    ".doc": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".csv": "spreadsheet",
    ".tsv": "spreadsheet",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".ods": "spreadsheet",
    ".numbers": "spreadsheet",
    ".txt": "text",
    ".log": "text",
    ".rtf": "text",
    # code
    ".py": "code",
    ".pyi": "code",
    ".js": "code",
    ".mjs": "code",
    ".cjs": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".json": "code",
    ".jsonl": "code",
    ".yaml": "code",
    ".yml": "code",
    ".toml": "code",
    ".ini": "code",
    ".cfg": "code",
    ".sh": "code",
    ".bash": "code",
    ".zsh": "code",
    ".fish": "code",
    ".rs": "code",
    ".go": "code",
    ".java": "code",
    ".kt": "code",
    ".kts": "code",
    ".c": "code",
    ".h": "code",
    ".cc": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".cs": "code",
    ".swift": "code",
    ".rb": "code",
    ".php": "code",
    ".sql": "code",
    ".css": "code",
    ".scss": "code",
    ".sass": "code",
    ".vue": "code",
    ".svelte": "code",
    ".lua": "code",
    ".ex": "code",
    ".exs": "code",
    ".erl": "code",
    ".hs": "code",
    ".clj": "code",
    ".scala": "code",
    ".r": "code",
    ".pl": "code",
    ".ipynb": "code",
    ".dockerfile": "code",
    ".makefile": "code",
}

#: MIME table used when the name has no useful suffix (an attachment digest,
#: a browser download with a query string) and for the content route's
#: ``Content-Type``. Longest/most specific first is unnecessary: exact keys.
_MIME_KIND: Final = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/svg+xml": "image",
    "image/bmp": "image",
    "image/tiff": "image",
    "image/heic": "image",
    "image/avif": "image",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mp4": "audio",
    "audio/aac": "audio",
    "audio/flac": "audio",
    "audio/ogg": "audio",
    "audio/opus": "audio",
    "video/mp4": "video",
    "video/quicktime": "video",
    "video/webm": "video",
    "video/x-matroska": "video",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "pptx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "text/markdown": "markdown",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/csv": "spreadsheet",
    "text/tab-separated-values": "spreadsheet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "spreadsheet",
    "application/vnd.ms-excel": "spreadsheet",
    "text/plain": "text",
    "application/json": "code",
    "application/javascript": "code",
    "text/javascript": "code",
    "text/x-python": "code",
    "application/x-python": "code",
}

#: Extension -> MIME for the content route. A record that already carries a
#: producer-supplied content type keeps that instead.
_EXTENSION_MIME: Final = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
    ".heic": "image/heic",
    ".avif": "image/avif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mdx": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".rtf": "application/rtf",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".cjs": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".jsx": "text/javascript",
    ".css": "text/css",
    ".sh": "application/x-sh",
    ".bash": "application/x-sh",
    ".zsh": "application/x-sh",
    ".sql": "application/sql",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".xml": "application/xml",
}


@dataclass(frozen=True, slots=True)
class Classification:
    """The three derived answers. Immutable so a caller cannot mutate a cached row."""

    kind: str
    category: str
    content_type: str


def extension(name: str) -> str:
    """The lowercase suffix of a path or name, dot included; ``""`` when none."""
    base = os.path.basename(name or "")
    # ``Dockerfile``/``Makefile`` are handled as bare names below.
    _, suffix = os.path.splitext(base)
    return suffix.lower()


def is_known_extension(name: str) -> bool:
    """True for a suffix the Library can classify. The mention pass uses this.

    Bare, well-known code filenames are admitted too: ``Dockerfile`` has no
    suffix but is unambiguously code, and a user naming it in prose is naming
    a real file.
    """
    base = os.path.basename(name or "").lower()
    if base in _BARE_NAMES:
        return True
    return extension(name) in _EXTENSIONS


_BARE_NAMES: Final = {
    "dockerfile",
    "makefile",
    "justfile",
    "rakefile",
    "procfile",
    "gemfile",
    "podfile",
}


def classify(name: str, mime: str | None = None) -> Classification:
    """Classify one asset.

    Suffix wins over MIME when both are known: a ``.md`` file served as
    ``text/plain`` is still a markdown document to every renderer, and a
    generated image whose producer omitted a MIME type is still an image.
    MIME is the fallback for digest-only artifacts and suffix-less downloads.
    """
    base = os.path.basename(name or "").lower()
    kind = _EXTENSIONS.get(extension(base))
    if kind is None and base in _BARE_NAMES:
        kind = "code"
    if kind is None and mime:
        kind = _MIME_KIND.get(mime.split(";", 1)[0].strip().lower())
    if kind is None:
        kind = "other"
    return Classification(
        kind=kind,
        category=_KIND_CATEGORY[kind],
        content_type=content_type(name, mime),
    )


def content_type(name: str, mime: str | None = None) -> str:
    """The MIME a content route should answer with."""
    if mime:
        return mime.split(";", 1)[0].strip() or "application/octet-stream"
    guessed = _EXTENSION_MIME.get(extension(name))
    if guessed:
        return guessed
    base = os.path.basename(name or "").lower()
    if base in _BARE_NAMES:
        return "text/plain"
    return "application/octet-stream"


def category_of(kind: str) -> str:
    """The sidebar bucket for a kind; ``other`` for an unknown kind."""
    return _KIND_CATEGORY.get(kind, "other")


__all__ = [
    "CATEGORIES",
    "KINDS",
    "Classification",
    "category_of",
    "classify",
    "content_type",
    "extension",
    "is_known_extension",
]
