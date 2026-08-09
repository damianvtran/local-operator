"""Progressive-disclosure URL resolution for skills and harness guides.

``skill://`` and ``guide://`` intentionally share one containment-checked
resolver. Both protocols return a small markdown body only after the model
chooses to read a semantically surfaced name; reference paths remain lazy.
Wrappers keep the public ``resolve_skill_url`` contract stable while packaged
guides use :func:`resolve_resource_url` with their own scheme and error labels.

Resolution rejects absolute paths, traversal, symlink escapes, and dotfiles.
Directory listings are deterministic and bounded, while file reads consume at
most 200 KiB plus one byte so a large file cannot become a large allocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

from local_operator.skills.discovery import Skill

MAX_READ_BYTES = 200 * 1024
"""Files larger than this are truncated; resources ship text, not binaries."""

_MAX_LISTING_ENTRIES = 500
"""Directory listings never exceed this many entries (RS-14); the overflow is
summarized with a marker rather than streamed into context."""

_TRUNCATION_MARKER = "\n\n[... content truncated at 200KB ...]"


def _read_text_capped(path: Path) -> str:
    """Read a file as UTF-8 (lossy), truncating at MAX_READ_BYTES.

    The cap bounds the READ, not just the returned string: we read at most
    ``MAX_READ_BYTES + 1`` bytes so a multi-GB resource file is never loaded
    fully before being cut.
    """
    with path.open("rb") as fh:
        raw = fh.read(MAX_READ_BYTES + 1)
    if len(raw) > MAX_READ_BYTES:
        return raw[:MAX_READ_BYTES].decode("utf-8", errors="replace") + _TRUNCATION_MARKER
    return raw.decode("utf-8", errors="replace")


def _list_directory(directory: Path) -> str:
    """Render a directory listing: ``name/ (dir)`` for dirs, plain names for
    files, deterministic alphabetical order. Dotfiles are skipped (they are
    unlisted and unreadable by design) and the listing is capped at
    ``_MAX_LISTING_ENTRIES`` with an overflow marker."""
    lines: list[str] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.name.lower(), p.name))
    except OSError:
        return "(unreadable directory)"
    shown = 0
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if shown >= _MAX_LISTING_ENTRIES:
            break
        lines.append(f"{entry.name}/ (dir)" if entry.is_dir() else entry.name)
        shown += 1
    hidden = sum(1 for entry in entries if not entry.name.startswith(".")) - shown
    if hidden > 0:
        lines.append(f"[... {hidden} more entries not shown ...]")
    return "\n".join(lines) if lines else "(empty directory)"


def _resolve_child(
    resource: Skill,
    raw_path: str,
    *,
    scheme: str,
    label: str,
) -> str:
    """Join a URL child path under one resource with containment checks."""
    relative = unquote(raw_path.lstrip("/"))

    segments = relative.split("/")
    if relative.startswith("/") or any(part == ".." for part in segments):
        raise ValueError(
            f"Invalid {label} path '{raw_path}': absolute paths and '..' segments "
            "are not allowed"
        )
    if any(part.startswith(".") and part != "." for part in segments):
        raise ValueError(
            f"Invalid {label} path '{raw_path}': dotfiles are not listed and cannot be read"
        )

    base = resource.base_dir.resolve()
    target = (resource.base_dir / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Invalid {label} path '{raw_path}': escapes the {label} directory")
    if not target.exists():
        raise ValueError(f"{label.title()} path not found: {scheme}://{resource.name}/{relative}")
    if target.is_dir():
        return _list_directory(target)
    return _read_text_capped(target)


def resolve_resource_url(
    url: str,
    resources: Mapping[str, Skill],
    *,
    scheme: str,
    label: str,
) -> str | None:
    """Resolve one internal resource scheme, or return ``None`` for others.

    Unknown names list only resources in this scheme. That bounded error is the
    model's self-correction path and never leaks user skills while resolving a
    packaged guide (or vice versa).
    """
    prefix = f"{scheme}://"
    if not url.startswith(prefix):
        return None

    parts = urlsplit(url)
    name = unquote(parts.netloc)
    title = label.title()
    if not name:
        raise ValueError(f"{title} URL missing a name: expected {scheme}://<name>")

    resource = resources.get(name)
    if resource is None:
        available = ", ".join(sorted(resources.keys())) or "(none)"
        raise ValueError(f"Unknown {label}: {name}\nAvailable: {available}")

    path = parts.path
    if not path or not path.strip("/"):
        return _read_text_capped(resource.file_path)
    return _resolve_child(resource, path, scheme=scheme, label=label)


def resolve_skill_url(url: str, skills: Mapping[str, Skill]) -> str | None:
    """Resolve a ``skill://`` URL while preserving the public skills API."""
    return resolve_resource_url(url, skills, scheme="skill", label="skill")
